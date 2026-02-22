

# USAGE 

Run with this version to get data about 

* **top 25 CUDA operations** by total GPU time.
* Prints **top 15 CPU operations** by CPU time.
* Prints **top 15 memory-hungry operations** (GPU memory).
* a **Wall-clock time table** for important functions 
  - Note : for insight of where is the function in the code search by name in the world_model_interaction.py file and you will see when the timers is initialized and stopped )

 you can just see it in the logs or run with SAVE_TRACE    = True  to export the .json file with the metrics 
then in the log you will see 

Chrome trace saved → {TRACE_PATH}
Open at: https://ui.perfetto.dev

And you can observe the metrics through that page  



## Profiler Overview Line by line (made by cloude)

This file uses **two profiling approaches** working together:

1. **PyTorch Profiler** (torch.profiler) - low-level GPU/CPU operation tracking
2. **Custom Timer class** - wall-clock timing for high-level code sections

------

## Part 1: Imports and Timer Class ([lines 26-65](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html))

### Line 27: Import PyTorch Profiler

- [profile](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html): Context manager that collects performance stats
- [record_function](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html): Labels specific code blocks for tracking
- [ProfilerActivity](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html): Enum to select CPU/CUDA tracking

### Lines 35-50: Custom Timer Class

**Global dictionary** storing timing measurements. Each key is a label, value is a list of elapsed times.

Creates a context manager (used with `with` statement) that you can name.

- **[torch.cuda.synchronize()](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)**: **CRITICAL** - waits for all GPU operations to finish. Without this, timing would be inaccurate because GPU ops are asynchronous.
- **[time.perf_counter()](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)**: High-resolution timer, starts the clock.

- Syncs GPU again to ensure all work completed
- Calculates elapsed time
- Appends to the global [_TIMINGS](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) dict under this label

### Lines 53-67: Print Summary Function

- Averages all measurements per label
- Sorts by time (descending)
- Shows **percentage** of total time and **count** of measurements

------

## Part 2: Configuration ([lines 324-331](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html))

**Why these values?**

- **[WARMUP_ITERS=1](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)**: Skip first iteration because it includes model loading/JIT compilation
- **[PROFILE_ITERS=2](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)**: Only profile 2 iterations to avoid memory overflow (profiler stores EVERY kernel call)
- **[SAVE_TRACE=False](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)**: Chrome trace files can be **gigabytes** in size, disabled by default

------

## Part 3: Profiler Setup ([lines 418-435](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html))

Track both CPU and GPU operations.

- **[record_shapes=True](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)**: Stores tensor shapes for each operation
- **[profile_memory=True](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)**: Tracks GPU memory allocation/deallocation

**This is the scheduler that controls WHEN profiling happens:**

| Iteration | What happens                                                 |
| --------- | ------------------------------------------------------------ |
| 0         | **WAIT** - profiler does nothing                             |
| 1         | **WARMUP** - flushes JIT cache (data not recorded)           |
| 2-3       | **ACTIVE** - actually collecting profiling data              |
| 4+        | Profiling stopped ([repeat=1](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) means one cycle only) |

Normally called after each profiling cycle, but we handle printing manually.

Starts the profiler and sets a flag to prevent multiple prints.

------

## Part 4: Main Loop Usage ([lines 453-571](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html))

### Decision-Making Pass ([lines 453-467](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html))

**Nested profiling**:

- [record_function("DM_pass")](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html): Labels this in PyTorch profiler output
- Outer [Timer](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html): Measures the entire decision-making pipeline (wall-clock)
- Inner [Timer](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html): Measures just the synthesis function

### World Model Pass ([lines 491-503](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html))

Same pattern - measures world model interaction separately.

### Queue Updates & I/O ([lines 506-542](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html))

Tracks overhead from data management and I/O operations.

### Profiler Step & Results ([lines 545-572](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html))

**CRITICAL**: Advances the profiler schedule by one step. Without this, the scheduler won't progress through wait→warmup→active phases.

After iteration 3 (0-indexed), stops profiling.

Prints **top 25 CUDA operations** by total GPU time.

Prints **top 15 CPU operations** by CPU time.

Prints **top 15 memory-hungry operations** (GPU memory).

Saves JSON trace file viewable in Chrome's `chrome://tracing` or [https://ui.perfetto.dev](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) for detailed visualization.

Prints the custom [Timer](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) measurements.

------

## What is Being Measured?

### PyTorch Profiler measures:

- **Every CUDA kernel** (matrix multiply, convolution, activation, etc.)
- **CPU operations** (data loading, preprocessing)
- **Memory allocations** per operation
- **Tensor shapes** used

### Custom Timer measures:

- **Wall-clock time** for:
  - Full decision-making pass
  - Full world-model pass
  - Image synthesis separately
  - Queue updates
  - File I/O (TensorBoard + video writes)

This dual approach gives both **low-level kernel insights** (PyTorch profiler) and **high-level pipeline breakdown** (Timer class).