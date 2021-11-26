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