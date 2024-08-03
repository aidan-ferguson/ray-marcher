# C++ Ray-marcher

Simple C++ ray-marcher that outputs `.png` files.
 - Has random super-sampling
 - Multi-threading (went from 2:16 -> 0:14 on my machine)
 - Shadows
 - Camera animation support
 - Animations of 4D SDFs by taking 3D slices 

![samples/1920x1080.png](samples/1920x1080.png)
![samples/1440p-colour.png](samples/1440p-colour.png)

There are videos in the `samples` sub-directory along with some more images.

Compile with
```
g++ main.cpp -Ilib/glm
```

You can use the following FFMPEG command to create videos from the frames:
```
ffmpeg -framerate 30 -i ./frames/%d.png -c:v libx264 -pix_fmt yuv420p output.mp4
```

### CUDA

There's also a separate (unoptimised) CUDA implementation. Compared to the CPU multi-threaded version that takes 14 seconds, this renders a frame in under a second (RTX 4060 Laptop GPU).

Compile with:
```
nvcc main.cu -Ilib/stb -lcurand
```