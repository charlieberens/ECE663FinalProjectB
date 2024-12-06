## Images
All images are stored in `/usr/xtmp/cjb131/output_images/2`

`original` - Unedited Source Images

`signed` - Image with a watermark and that's it

`signed_compressed_XX` - Image with a watermark compressed to quailty XX%

`unsigned_compressed_XX` - Image with no watermark compressed to quailty XX%.

Images are saved as torch tensors. I wasn't confident that the method I was using to save them as images was saving them uncompressed.

## Visualizations
* I think it would be useful to have a histrogram like we had before w/ the distribution of the distance between an image; its signed, compressed signed, and compressed unsigned variants; and differing images. I don't think we should do any dimensionality reduction, just do it in the 1000 dimension latent space.

## Reference Code
The reference code opens each type of image, calculates the embeddings, and displays the L2s between them and the original image. It does not handle L2s between non-matching images.