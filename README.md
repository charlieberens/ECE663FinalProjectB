## File Structure
Up one directory there is a "data" folder `/home/users/cjb131/school/SigningDemo/data`. 

```
SigningDemo
 └─── data
 └─── HiDDeN
```

## Encoding Methods
### Alternate
The watermark is embedded in every nth pixel of the image. `n` is specified by the `--masking-args` argument.

Ex: 
```sh
python main.py new --data-dir ../data/midjourney/ -b 2 -e 300 --name alternate_32 --size 512 -m 32 --hash-mode screen --masking-args 2
```

This will embed a 32 bit mask in every other bit (every 2 bits) of a 512x512 image.

### Bitwise
The watermark is embedded in the last n bits of each pixel. `n` is specified by the `--masking-args` argument.

#### BitwiseA
The encoder takes the full image as input, zeros out the last n bits, then learns a number between 0 and 1 indicating how large the last few bits should be. 0 corresponds to xxxx0000. 1 corresponds to xxxx1111. There's everything else in between. 

This does not work particularly well.

#### BitwiseB
The encoder takes the last n bits as input, and outputs a number between 0 and 1 that is similarly converted to n bits. This works much better.

#### BitwiseC
This is the same as bitwiseB but durring the "after concat layer", the model additionally has access to the original image.

> Dropdown Test
WAHSIKD KASD