# Fruit-Experimental: An extension to the networks from Fruits-360 repository #

The code here relies on the images from the [Fruits-360](https://github.com/Horea94/Fruit-Images-Dataset) repository so you will need to download that as well.

# Running Object Detection

## 1 Prerequisites
### 1.1 Get dataset
[The Fruits-360 Repository](https://github.com/Horea94/Fruit-Images-Dataset)
I recommend creating soft links to the Training and Test folders from the Fruits-360 repo in the folder where you download Fruit-Experimental.
Open a command line or terminal in the Fruit-Experimental and do:

`mklink /D Training "path-to-Fruits-360/Training"`

`mklink /D Test "path-to-Fruits-360/Test"`

Where "path-to-Fruits-360" is the absolute or relative path to the Fruits-360 folder, which you should have after downloading the Fruits-360 repo.

### 1.2 Install  Python, TensorFlow 2 and other dependencies

[Install Python 3.8](https://www.python.org/downloads/) *TensorFlow does not support a higher version*

[Install TensorFlow 2](https://www.tensorflow.org/install)

Install opencv by running in a command line:

`pip install opencv-python`

Install scikit-learn by running in a command line:

`pip install scikit-learn`

## 2 Config (in folder src/object_detection)

Make a copy of each of the **detection_config.py.tmplt** and **labels.txt.tmplt** files and name them **detection_config.py** and **labels.txt**, respectively.
The **labels.txt** file is used to select the source classes from the Fruits-360 dataset to build the detection dataset.
The **detection_config.py** file contains all the settings related to the data generation and network configuration.

## 3 Build dataset
## 4 Train
## 5 Test

## License ##

MIT License

Copyright (c) 2017-2020 [Mihai Oltean](https://mihaioltean.github.io), Horea Muresan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
