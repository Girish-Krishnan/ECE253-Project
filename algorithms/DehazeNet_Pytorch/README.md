# DehazeNet_Pytorch
A Pytorch implementation for DehazeNet in paper 'DehazeNet: An End-to-End System for Single Image Haze Removal'

@article{cai2016dehazenet,  
	author = {Bolun Cai, Xiangmin Xu, Kui Jia, Chunmei Qing and Dacheng Tao},  
	title={DehazeNet: An End-to-End System for Single Image Haze Removal},  
	journal={IEEE Transactions on Image Processing},  
	year={2016},   
	volume={25},   
	number={11},   
	pages={5187-5198},  
	}
  
Run create_dataset.py to create a training dataset.   
Run DehazeNet-pytorch.py.train() to train a model.   
Run DehazeNet-pytorch.py.defog() to defog a picture.   
The model is trained on GPU and defog() is run on CPU.  

The model I trained didn't work well.  
In the paper, 'To refine the transmission map, guided image filtering [15] is used to smooth the image.', but I did't do it.

## Running model on a directory of images

The script `results.py` can be used to run the model on a directory of images.

Example usage:

```bash
python results.py --input_dir path/to/input/images --output_dir path/to/save/results --weights defog4_noaug.pth
```