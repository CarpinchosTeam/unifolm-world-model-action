import argparse, os, glob
import pandas as pd
import random
import torch
import torchvision
import h5py
import numpy as np
import logging
import einops
import warnings
import imageio
import time

from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict
from torch import nn
from eval_utils import populate_queues, log_to_tensorboard
from collections import deque
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# ─── Profiler imports ────────────────────────────────────────────────────────
from torch.profiler import profile, record_function, ProfilerActivity
# ─────────────────────────────────────────────────────────────────────────────

from unifolm_wma.models.samplers.ddim import DDIMSampler
from unifolm_wma.utils.utils import instantiate_from_config


# ─── Simple wall-clock timer dict, accumulated across steps ──────────────────
_TIMINGS: dict[str, list[float]] = {}

class Timer:
    """Context manager that accumulates wall-clock time by label."""
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        torch.cuda.synchronize()   # flush GPU before starting clock
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        torch.cuda.synchronize()   # flush GPU before stopping clock
        elapsed = time.perf_counter() - self._t0
        _TIMINGS.setdefault(self.label, []).append(elapsed)


def print_timing_summary():
    print("\n" + "=" * 60)
    print("  WALL-CLOCK TIMING SUMMARY (averaged over profiled iters)")
    print("=" * 60)
    total = 0.0
    rows = []
    for label, times in _TIMINGS.items():
        avg = sum(times) / len(times)
        total += avg
        rows.append((label, avg, len(times)))
    rows.sort(key=lambda x: -x[1])
    for label, avg, n in rows:
        pct = 100 * avg / total if total > 0 else 0
        print(f"  {label:<45} {avg:>7.3f}s  ({pct:>5.1f}%)  [n={n}]")
    print(f"  {'TOTAL':<45} {total:>7.3f}s")
    print("=" * 60 + "\n")
# ─────────────────────────────────────────────────────────────────────────────


def get_device_from_parameters(module: nn.Module) -> torch.device:
    return next(iter(module.parameters())).device


def write_video(video_path: str, stacked_frames: list, fps: int) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                "pkg_resources is deprecated as an API",
                                category=DeprecationWarning)
        imageio.mimsave(video_path, stacked_frames, fps=fps)


def get_filelist(data_dir: str, postfixes: list[str]) -> list[str]:
    patterns = [
        os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes
    ]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list


def load_model_checkpoint(model: nn.Module, ckpt: str) -> nn.Module:
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            new_pl_sd = OrderedDict()
            for k, v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]] = state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model


def is_inferenced(save_dir: str, filename: str) -> bool:
    video_file = os.path.join(save_dir, "samples_separate",
                              f"{filename[:-4]}_sample0.mp4")
    return os.path.exists(video_file)


def save_results(video: Tensor, filename: str, fps: int = 8) -> None:
    video = video.detach().cpu()
    video = torch.clamp(video.float(), -1., 1.)
    n = video.shape[0]
    video = video.permute(2, 0, 1, 3, 4)

    frame_grids = [
        torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0)
        for framesheet in video
    ]
    grid = torch.stack(frame_grids, dim=0)
    grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
    torchvision.io.write_video(filename,
                               grid,
                               fps=fps,
                               video_codec='h264',
                               options={'crf': '10'})


def get_init_frame_path(data_dir: str, sample: dict) -> str:
    rel_video_fp = os.path.join(sample['data_dir'],
                                str(sample['videoid']) + '.png')
    full_image_fp = os.path.join(data_dir, 'images', rel_video_fp)
    return full_image_fp


def get_transition_path(data_dir: str, sample: dict) -> str:
    rel_transition_fp = os.path.join(sample['data_dir'],
                                     str(sample['videoid']) + '.h5')
    full_transition_fp = os.path.join(data_dir, 'transitions',
                                      rel_transition_fp)
    return full_transition_fp


def prepare_init_input(start_idx: int,
                       init_frame_path: str,
                       transition_dict: dict[str, torch.Tensor],
                       frame_stride: int,
                       wma_data,
                       video_length: int = 16,
                       n_obs_steps: int = 2) -> dict[str, Tensor]:
    """
    Extracts a structured sample from a video sequence including frames, states, and actions,
    along with properly padded observations and pre-processed tensors for model input.

    Args:
        start_idx (int): Starting frame index for the current clip.
        video: decord video instance.
        transition_dict (Dict[str, Tensor]): Dictionary containing tensors for 'action', 
                                             'observation.state', 'action_type', 'state_type'.
        frame_stride (int): Temporal stride between sampled frames.
        wma_data: Object that holds configuration and utility functions like normalization, 
                transformation, and resolution info.
        video_length (int, optional): Number of frames to sample from the video. Default is 16.
        n_obs_steps (int, optional): Number of historical steps for observations. Default is 2.
    """

    indices = [start_idx + frame_stride * i for i in range(video_length)]
    init_frame = Image.open(init_frame_path).convert('RGB')
    init_frame = torch.tensor(np.array(init_frame)).unsqueeze(0).permute(
        3, 0, 1, 2).float()

    if start_idx < n_obs_steps - 1:
        state_indices = list(range(0, start_idx + 1))
        states = transition_dict['observation.state'][state_indices, :]
        num_padding = n_obs_steps - 1 - start_idx
        first_slice = states[0:1, :]  # (t, d)
        padding = first_slice.repeat(num_padding, 1)
        states = torch.cat((padding, states), dim=0)
    else:
        state_indices = list(range(start_idx - n_obs_steps + 1, start_idx + 1))
        states = transition_dict['observation.state'][state_indices, :]

    actions = transition_dict['action'][indices, :]
    ori_state_dim = states.shape[-1]
    ori_action_dim = actions.shape[-1]

    frames_action_state_dict = {
        'action': actions,
        'observation.state': states,
    }
    frames_action_state_dict = wma_data.normalizer(frames_action_state_dict)
    frames_action_state_dict = wma_data.get_uni_vec(
        frames_action_state_dict,
        transition_dict['action_type'],
        transition_dict['state_type'],
    )

    if wma_data.spatial_transform is not None:
        init_frame = wma_data.spatial_transform(init_frame)
    init_frame = (init_frame / 255 - 0.5) * 2

    data = {
        'observation.image': init_frame,
    }
    data.update(frames_action_state_dict)
    return data, ori_state_dim, ori_action_dim


def get_latent_z(model, videos: Tensor) -> Tensor:
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def preprocess_observation(model, observations):
    return_observations = {}
    if isinstance(observations["pixels"], dict):
        imgs = {
            f"observation.images.{key}": img
            for key, img in observations["pixels"].items()
        }
    else:
        imgs = {"observation.images.top": observations["pixels"]}

    for imgkey, img in imgs.items():
        img = torch.from_numpy(img)
        _, h, w, c = img.shape
        assert c < h and c < w, f"expect channel first images, but instead {img.shape}"

        # Sanity check that images are uint8
        assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

        # Convert to channel first of type float32 in range [0,1]
        img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
        img = img.type(torch.float32)
        return_observations[imgkey] = img

    return_observations["observation.state"] = torch.from_numpy(
        observations["agent_pos"]).float()
    return_observations['observation.state'] = model.normalize_inputs({
        'observation.state':
        return_observations['observation.state'].to(model.device)
    })['observation.state']
    return return_observations


def image_guided_synthesis_sim_mode(
        model, prompts, observation, noise_shape,
        action_cond_step=16, n_samples=1, ddim_steps=50,
        ddim_eta=1.0, unconditional_guidance_scale=1.0,
        fs=None, text_input=True, timestep_spacing='uniform',
        guidance_rescale=0.0, sim_mode=True, **kwargs):

    b, _, t, _, _ = noise_shape
    ddim_sampler = DDIMSampler(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    # ── Conditioning build-up ─────────────────────────────────────────────────
    with record_function("cond/image_embedding"):
        img = observation['observation.images.top'].permute(0, 2, 1, 3, 4)
        cond_img = rearrange(img, 'b o c h w -> (b o) c h w')[-1:]
        cond_img_emb = model.embedder(cond_img)
        cond_img_emb = model.image_proj_model(cond_img_emb)

    with record_function("cond/vae_encode"):
        if model.model.conditioning_key == 'hybrid':
            z = get_latent_z(model, img.permute(0, 2, 1, 3, 4))
            img_cat_cond = z[:, :, -1:, :, :]
            img_cat_cond = repeat(img_cat_cond,
                                  'b c t h w -> b c (repeat t) h w',
                                  repeat=noise_shape[2])
            cond = {"c_concat": [img_cat_cond]}

    with record_function("cond/text_conditioning"):
        if not text_input:
            prompts = [""] * batch_size
        cond_ins_emb = model.get_learned_conditioning(prompts)

    with record_function("cond/state_action_projection"):
        cond_state_emb = model.state_projector(observation['observation.state'])
        cond_state_emb = cond_state_emb + model.agent_state_pos_emb
        cond_action_emb = model.action_projector(observation['action'])
        cond_action_emb = cond_action_emb + model.agent_action_pos_emb
        if not sim_mode:
            cond_action_emb = torch.zeros_like(cond_action_emb)

    cond["c_crossattn"] = [
        torch.cat(
            [cond_state_emb, cond_action_emb, cond_ins_emb, cond_img_emb],
            dim=1)
    ]
    cond["c_crossattn_action"] = [
        observation['observation.images.top'][:, :, -model.n_obs_steps_acting:],
        observation['observation.state'][:, -model.n_obs_steps_acting:],
        sim_mode,
        False,
    ]

    uc = None
    kwargs.update({"unconditional_conditioning_img_nonetext": None})

    # ── DDIM sampling (the expensive part) ───────────────────────────────────
    with record_function("ddim_sampling"):
        samples, actions, states, intermedia = ddim_sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=batch_size,
            shape=noise_shape[1:],
            verbose=False,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            cfg_img=None,
            mask=None,
            x0=None,
            fs=fs,
            timestep_spacing=timestep_spacing,
            guidance_rescale=guidance_rescale,
            **kwargs)

    # ── VAE decode ────────────────────────────────────────────────────────────
    with record_function("vae_decode"):
        batch_images = model.decode_first_stage(samples)

    return batch_images, actions, states


def run_inference(args: argparse.Namespace, gpu_num: int, gpu_no: int) -> None:
    # ── Profiling config ──────────────────────────────────────────────────────
    # We profile only PROFILE_ITERS iterations then print stats and continue
    # normally (no trace saved to disk to avoid OOM).
    PROFILE_ITERS = 2          # how many iters to profile (keep small!)
    WARMUP_ITERS  = 1          # iters to skip before profiling starts
    SAVE_TRACE    = True      # set True only if you have plenty of disk/RAM
    TRACE_PATH    = "./profiler_trace.json"   # only used if SAVE_TRACE=True
    # ─────────────────────────────────────────────────────────────────────────

    os.makedirs(args.savedir + '/inference', exist_ok=True)
    log_dir = args.savedir + f"/tensorboard"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    csv_path = os.path.join(args.prompt_dir, f"{args.dataset}.csv")
    df = pd.read_csv(csv_path)

    config = OmegaConf.load(args.config)
    config['model']['params']['wma_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(config.model)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()
    print(f'>>> Load pre-trained model ...')

    logging.info("***** Configing Data *****")
    data = instantiate_from_config(config.data)
    data.setup()
    print(">>> Dataset is successfully loaded ...")

    model = model.cuda(gpu_no)
    device = get_device_from_parameters(model)

    assert (args.height % 16 == 0) and (args.width % 16 == 0)
    assert args.bs == 1

    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'>>> Generate {n_frames} frames per generation ...')
    noise_shape = [args.bs, channels, n_frames, h, w]

    for idx in range(0, len(df)):
        sample = df.iloc[idx]
        init_frame_path = get_init_frame_path(args.prompt_dir, sample)
        ori_fps = float(sample['fps'])

        video_save_dir = args.savedir + f"/inference/sample_{sample['videoid']}"
        os.makedirs(video_save_dir, exist_ok=True)
        os.makedirs(video_save_dir + '/dm', exist_ok=True)
        os.makedirs(video_save_dir + '/wm', exist_ok=True)

        transition_path = get_transition_path(args.prompt_dir, sample)
        with h5py.File(transition_path, 'r') as h5f:
            transition_dict = {}
            for key in h5f.keys():
                transition_dict[key] = torch.tensor(h5f[key][()])
            for key in h5f.attrs.keys():
                transition_dict[key] = h5f.attrs[key]

        for fs in args.frame_stride:
            sample_save_dir = f'{video_save_dir}/dm/{fs}'
            os.makedirs(sample_save_dir, exist_ok=True)
            sample_save_dir = f'{video_save_dir}/wm/{fs}'
            os.makedirs(sample_save_dir, exist_ok=True)
            wm_video = []

            cond_obs_queues = {
                "observation.images.top": deque(maxlen=model.n_obs_steps_imagen),
                "observation.state":      deque(maxlen=model.n_obs_steps_imagen),
                "action":                 deque(maxlen=args.video_length),
            }

            start_idx = 0
            model_input_fs = ori_fps // fs
            batch, ori_state_dim, ori_action_dim = prepare_init_input(
                start_idx, init_frame_path, transition_dict, fs,
                data.test_datasets[args.dataset],
                n_obs_steps=model.n_obs_steps_imagen)

            observation = {
                'observation.images.top':
                    batch['observation.image'].permute(1, 0, 2, 3)[-1].unsqueeze(0),
                'observation.state':
                    batch['observation.state'][-1].unsqueeze(0),
                'action':
                    torch.zeros_like(batch['action'][-1]).unsqueeze(0)
            }
            observation = {
                k: v.to(device, non_blocking=True) for k, v in observation.items()
            }
            cond_obs_queues = populate_queues(cond_obs_queues, observation)

            # ── Build the profiler (only captures PROFILE_ITERS active steps) ──
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                # wait=warmup, warmup=1 (jit flush), active=profile, repeat=1 then stops
                schedule=torch.profiler.schedule(
                    wait=WARMUP_ITERS,
                    warmup=1,
                    active=PROFILE_ITERS,
                    repeat=1,
                ),
                # on_trace_ready fires once per (wait+warmup+active) cycle
                on_trace_ready=None,   # we print manually below
            )
            prof.start()
            profiling_done = False
            # ─────────────────────────────────────────────────────────────────

            for itr in tqdm(range(args.n_iter)):

                observation = {
                    'observation.images.top':
                        torch.stack(list(cond_obs_queues['observation.images.top']), dim=1)
                             .permute(0, 2, 1, 3, 4),
                    'observation.state':
                        torch.stack(list(cond_obs_queues['observation.state']), dim=1),
                    'action':
                        torch.stack(list(cond_obs_queues['action']), dim=1),
                }
                observation = {
                    k: v.to(device, non_blocking=True) for k, v in observation.items()
                }

                # ── Decision-making pass ──────────────────────────────────────
                print(f'>>> Step {itr}: generating actions ...')
                with record_function("DM_pass"):
                    with Timer("DM: full decision-making pass"):
                        with Timer("DM: image_guided_synthesis"):
                            pred_videos_0, pred_actions, _ = image_guided_synthesis_sim_mode(
                                model, sample['instruction'], observation,
                                noise_shape,
                                action_cond_step=args.exe_steps,
                                ddim_steps=args.ddim_steps,
                                ddim_eta=args.ddim_eta,
                                unconditional_guidance_scale=args.unconditional_guidance_scale,
                                fs=model_input_fs,
                                timestep_spacing=args.timestep_spacing,
                                guidance_rescale=args.guidance_rescale,
                                sim_mode=False)

                # ── Update action queues ──────────────────────────────────────
                with record_function("queue_update_actions"):
                    for aidx in range(len(pred_actions[0])):
                        obs_a = {'action': pred_actions[0][aidx:aidx + 1]}
                        obs_a['action'][:, ori_action_dim:] = 0.0
                        cond_obs_queues = populate_queues(cond_obs_queues, obs_a)

                observation = {
                    'observation.images.top':
                        torch.stack(list(cond_obs_queues['observation.images.top']), dim=1)
                             .permute(0, 2, 1, 3, 4),
                    'observation.state':
                        torch.stack(list(cond_obs_queues['observation.state']), dim=1),
                    'action':
                        torch.stack(list(cond_obs_queues['action']), dim=1),
                }
                observation = {
                    k: v.to(device, non_blocking=True) for k, v in observation.items()
                }

                # ── World-model interaction pass ──────────────────────────────
                print(f'>>> Step {itr}: interacting with world model ...')
                with record_function("WM_pass"):
                    with Timer("WM: full world-model pass"):
                        with Timer("WM: image_guided_synthesis"):
                            pred_videos_1, _, pred_states = image_guided_synthesis_sim_mode(
                                model, "", observation, noise_shape,
                                action_cond_step=args.exe_steps,
                                ddim_steps=args.ddim_steps,
                                ddim_eta=args.ddim_eta,
                                unconditional_guidance_scale=args.unconditional_guidance_scale,
                                fs=model_input_fs,
                                text_input=False,
                                timestep_spacing=args.timestep_spacing,
                                guidance_rescale=args.guidance_rescale)

                # ── Update obs queues from world-model output ─────────────────
                with record_function("queue_update_obs"):
                    with Timer("queue: update from WM output"):
                        for eidx in range(args.exe_steps):
                            obs_wm = {
                                'observation.images.top':
                                    pred_videos_1[0][:, eidx:eidx + 1].permute(1, 0, 2, 3),
                                'observation.state':
                                    torch.zeros_like(pred_states[0][eidx:eidx + 1])
                                    if args.zero_pred_state
                                    else pred_states[0][eidx:eidx + 1],
                                'action':
                                    torch.zeros_like(pred_actions[0][-1:])
                            }
                            obs_wm['observation.state'][:, ori_state_dim:] = 0.0
                            cond_obs_queues = populate_queues(cond_obs_queues, obs_wm)

                # ── Save results ──────────────────────────────────────────────
                with record_function("save_results"):
                    with Timer("IO: tensorboard + video save"):
                        sample_tag = f"{args.dataset}-vid{sample['videoid']}-dm-fs-{fs}/itr-{itr}"
                        log_to_tensorboard(writer, pred_videos_0, sample_tag, fps=args.save_fps)
                        sample_tag = f"{args.dataset}-vid{sample['videoid']}-wd-fs-{fs}/itr-{itr}"
                        log_to_tensorboard(writer, pred_videos_1, sample_tag, fps=args.save_fps)

                        save_results(pred_videos_0.cpu(),
                                     f'{video_save_dir}/dm/{fs}/itr-{itr}.mp4',
                                     fps=args.save_fps)
                        save_results(pred_videos_1.cpu(),
                                     f'{video_save_dir}/wm/{fs}/itr-{itr}.mp4',
                                     fps=args.save_fps)

                wm_video.append(pred_videos_1[:, :, :args.exe_steps].cpu())

                # ── Advance profiler schedule ─────────────────────────────────
                prof.step()

                # After (WARMUP_ITERS + 1 + PROFILE_ITERS) total steps, print and stop
                _profile_trigger = WARMUP_ITERS + 1 + PROFILE_ITERS
                if itr == _profile_trigger - 1 and not profiling_done:
                    prof.stop()
                    profiling_done = True

                    print("\n" + "=" * 60)
                    print("  TORCH PROFILER — TOP CUDA OPS")
                    print("=" * 60)
                    print(prof.key_averages().table(
                        sort_by="cuda_time_total",
                        row_limit=25,
                    ))

                    print("\n" + "=" * 60)
                    print("  TORCH PROFILER — TOP CPU OPS")
                    print("=" * 60)
                    print(prof.key_averages().table(
                        sort_by="cpu_time_total",
                        row_limit=15,
                    ))

                    print("\n" + "=" * 60)
                    print("  TORCH PROFILER — MEMORY (GPU)")
                    print("=" * 60)
                    print(prof.key_averages().table(
                        sort_by="self_cuda_memory_usage",
                        row_limit=15,
                    ))

                    if SAVE_TRACE:
                        prof.export_chrome_trace(TRACE_PATH)
                        print(f"\n  Chrome trace saved → {TRACE_PATH}")
                        print("  Open at: https://ui.perfetto.dev\n")

                    print_timing_summary()

                print('>' * 24)

            full_video = torch.cat(wm_video, dim=2)
            sample_tag = f"{args.dataset}-vid{sample['videoid']}-wd-fs-{fs}/full"
            log_to_tensorboard(writer, full_video, sample_tag, fps=args.save_fps)
            save_results(full_video,
                         f"{video_save_dir}/../{sample['videoid']}_full_fs{fs}.mp4",
                         fps=args.save_fps)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--config", type=str)
    parser.add_argument("--prompt_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frame_stride", type=int, nargs='+', required=True)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--video_length", type=int, default=16)
    parser.add_argument("--num_generation", type=int, default=1)
    parser.add_argument("--timestep_spacing", type=str, default="uniform")
    parser.add_argument("--guidance_rescale", type=float, default=0.0)
    parser.add_argument("--perframe_ae", action='store_true', default=False)
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--exe_steps", type=int, default=16)
    parser.add_argument("--n_iter", type=int, default=40)
    parser.add_argument("--zero_pred_state", action='store_true', default=False)
    parser.add_argument("--save_fps", type=int, default=8)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    seed = args.seed
    if seed < 0:
        seed = random.randint(0, 2**31)
    seed_everything(seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)