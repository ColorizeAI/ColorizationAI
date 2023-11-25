#!/bin/bash

# Update package list
apt-get update

# Install required dependencies
apt-get install -y ffmpeg libsm6 libxext6
