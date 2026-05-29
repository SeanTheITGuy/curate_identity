# curate_identity
## Simple application for testing and sorting images based on facial similarity.

This was created to allow for evolutionary generation of character LoRA training. Its is model agnostic, but I usually work with Z Image Turbo.

A first generation LoRA can be created by creating a long, detailed description of a character, including facial features, hair style, body type, etc. As much detail as possible to try to lock in a character. Then, fix the random seed, and generate a large number of potential images. I have included the randomish three stage Z Image Turbo ComfyUI workflow I use. It generates decent quality, high realism images. It's a bit slow, given the detail I strive for, but if you don't require the detail, you can always bypass the third and possibly second sampler stages.

Generate as many as you have the time for. 1000 generations might only produce 50-100 "good" matches. Manually sort through the images, pick 2-3 that you like and have a visual similarity. Put those in the `refs/` directory. put the remainder in the `gens/` directory. Run the application

`python ./curate_identity.py --refs refs/ --input gens/ --output sorted/ --dry-run`

On first run it will download the models required, then it will create an identity centroid (math model) based on the reference images you selected. It will then scan all the images in `gens/` and give you a list of their rankings of facial similarity compared to the centroid.  From there you can get an idea of the number of images above the default 82% ranking (if any). For this first round, you'll likely need to set a approve/review threshold to get a reasonable number of approved images. Pick a value that will give you somewhere around 10% of your total number approved and around 2-5% further to review.

```
=== DRY RUN RESULTS ===

0.8126 [REVIEW] gens/CUDA_00164_.png
0.7518 [REVIEW] gens/CUDA_00166_.png
0.7347 [REJECT] gens/CUDA_00168_.png
0.7316 [REJECT] gens/CUDA_00186_.png
0.7184 [REJECT] gens/CUDA_00162_.png
0.7120 [REJECT] gens/CUDA_00177_.png
0.7037 [REJECT] gens/CUDA_00181_.png
0.6913 [REJECT] gens/CUDA_00163_.png
0.6835 [REJECT] gens/CUDA_00183_.png
0.6833 [REJECT] gens/CUDA_00167_.png
0.6740 [REJECT] gens/CUDA_00170_.png
0.6736 [REJECT] gens/CUDA_00190_.png
0.6681 [REJECT] gens/CUDA_00173_.png
0.6680 [REJECT] gens/CUDA_00174_.png
0.6632 [REJECT] gens/CUDA_00178_.png
0.6493 [REJECT] gens/CUDA_00187_.png
0.6461 [REJECT] gens/CUDA_00185_.png
0.6395 [REJECT] gens/CUDA_00180_.png
0.6345 [REJECT] gens/CUDA_00182_.png
0.6053 [REJECT] gens/CUDA_00165_.png
0.6011 [REJECT] gens/CUDA_00176_.png
0.5990 [REJECT] gens/CUDA_00188_.png
0.5756 [REJECT] gens/CUDA_00169_.png
0.5755 [REJECT] gens/CUDA_00175_.png
0.5755 [REJECT] gens/CUDA_00172_.png
0.5723 [REJECT] gens/CUDA_00184_.png
0.5721 [REJECT] gens/CUDA_00171_.png
0.5703 [REJECT] gens/CUDA_00179_.png
0.5605 [REJECT] gens/CUDA_00189_.png
```

In this small example set, I would probably do something like:
`python ./curate_identity.py --refs refs/ --input gens/ --output sorted/ --approved-threshold 0.73 --review-threshold 0.7`

Removing the `--dry-run` will copy the `gens/` data into the `sorted/` directory appropriately. Review the "review" ones, and if they look similar enough to you, copy them to approved. Images flagged for "review" are often ones that match facial likeness, but are in a different position, framing, or partially obscured. These are good additions to a dataset.

Next, load up your favorite LoRA trainer and employ the new dataset you've created.

Once you have this v1 LoRA, repeat the process, enabling this new LoRA in the dataset generator workflow, and you'll find the images generated test as much higher likeness. I will usually generate a smaller number this time (200-300) and take everything over 82% (an arbitrary threshold that works for me) and create a v2 LoRA from that.  At that point you have a solid, stable, diverse character LoRA.


