**How to count lines in a document?**
`wc -l <filename>`

**Get a count of only the directories in the current directory**

`echo */ | wc`

2nd digit represents no. of directories.

`find . -type f -name "abc*" `
The above command will search the file that starts with abc under the current working directory.



**Add a new user to Unix machine**
`sudo adduser username && sudo usermod -aG sudo username`


**Remove PPA source from Terminal**
 Use the following command to see all the PPAs added in your system:

`sudo ls /etc/apt/sources.list.d`

Look for your desire PPA here and then remove the PPA using the following command:

`sudo rm -i /etc/apt/sources.list.d/PPA_Name.list`

**Install Python 3.x**

```python
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8
```


#### **Install CUDA 11.1 and other related problems**

1. First, you need to clean all previous installations.

```bash
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt  autoremove nvidia-cuda-toolkit
sudo apt autoremove nvidia-*
```

Ref: https://medium.com/@chami.soufiane/installation-guide-for-cuda-10-x-on-ubuntu-16-04-18-04-20-04-3a826a110ff5

2. Install CUDA follows the instructions from their homepage: https://developer.nvidia.com/cuda-11.1.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal

3. Stop X server  Ref: https://askubuntu.com/questions/149206/how-to-install-nvidia-run

4. Error: Could not load dynamic library 'libcudart.so.11.0' #45930

```bash
First, find out where the "libcudart.so.11.0" is
If you lost other at error stack, you can replace the "libcudart.so.11.0" by your word in below:

sudo find / -name 'libcudart.so.11.0'

Output in my system.This result shows where the "libcudart.so.11.0" is in my system:

/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudart.so.11.0
If the result shows nothing, please make sure you have install cuda or other staff that must install in your system.

Second, add the path to environment file.
edit /etc/profile
sudo vim /etc/profile
append path to "LD_LIBRARY_PATH" in profile file
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1/targets/x86_64-linux/lib
make environment file work
source /etc/profile
```

Ref: https://github.com/tensorflow/tensorflow/issues/45930


**Run Linux Commands Using the timeout Tool**

```bash
timeout [OPTION] DURATION COMMAND [ARG]...

```

Example;

```bash
timeout 5s ping google.com

```

Ref: https://www.tecmint.com/run-linux-command-with-time-limit-and-timeout/