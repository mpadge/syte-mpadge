## Syte coding exercise

> The primary performance goal is to minimize the response time of the
> get_image(lat, long, radius) function. Given that reading a single .jp2 file
> can take several seconds or minutes, your objective is to maximize the
> throughput.

The approach taken here is to use GDAL for all of the heavy lifting, through
reading only the required portions of each file, and through applying
aggregation (or down-sampling) filters to data _on disk_. This entirely
eliminates any need for these files to be directly read into memory at their
original resolution.

The repository is `makefile` controlled, and includes an initial `help` option.
`make check` will check for all required python dependencies, and provide
information on any missing ones. `make run` will then run the script. This does
not save any resultant image file, but generates a timing message indicating
that the whole procedure generally only takes a handful of seconds.

### Description of procedure

Input coordinates and a buffer distance are converted both to a bounding range
in EPSG:25832 coordinates, and a list of corresponding image files needed to
enclose these coordinates. The `aggregate_one_file` function then uses this
information to read only the required geographic portion of each file, and only
in aggregated form. The speed of the entire script is entirely due to the
routines applied in this function.

The `generate_merged_files` function is used to generate an initially merged
version of the input latitude, longitude, and buffer distance parameters. This
version will always have a somewhat higher resolution than the final required
output resolution, and so the function also applies a further aggregation or
downsampling routine to achieve the desired output resolution of 256x256.

### Error handling

Because of personal time restrictions, the current code contains no error
handling routines.

### Benchmarking

Benchmarking was performed, although is not included in current code. The
primary parameter influencing computation times is `buffer_distance`, which
must generally scale with `N^2`, while all other parameters must scale linearly
at most. Benchmarking demonstrating that responses to this parameter lie well
below quadratic, which is encouraging.

### Optimizations

**Downscale input images**

Potential optimizations that could be implemented depend very much on typical
envisioned application of the code, and in particular on how common
approximately repeated calls are likely to be. The entire performance
bottleneck is in the reading and aggregation of the image files, within
`aggregate_one_file`. Since the aim of this exercise was to generate 256x256
pixel output images from far higher resolution inputs, the single biggest
optimization step which could be implemented would be to initially
aggregate/downsample all input images to the smallest anticipated input buffer
size ("radius"). A buffer size of 100m translates to a reduction to under half
of the current height and width of these images. Such reductions would also
quadratically increase computational efficiency.

**Use local cache**

If further optimization were required, a second stage could involve a
coarsening of the initial `aggregate_one_file` function into discrete, coarser
chunks which could then be locally cached. An additional function could then be
written to align a precise set of `(lat, lon, buffer_size)` parameters to a
coarse file chunk, read that file from cache if it exists, and then trim to the
precisely required region. Details would again depend on anticipated usage
patterns, but chunking each image into, say, 10-by-10 smaller sections would
generate 100 possible sub-images from each input image, resulting in a maximum
of 6,400 locally-cached files. This cache would occupy a tiny fraction of the
size of the original data, and would likely enormously speed up the whole
routine.

**Parallelise**

The main `generate_merged_files` contains an initial `for` loop which occupies
most of the processing time. This ultimately calls `rasterio.read`, with
parameters to trim and aggregate the results. This function can also be called
in parallel mode, which would obviously also reduce computation times.
