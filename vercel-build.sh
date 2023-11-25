#!/bin/bash

# Ensure that Git LFS files are fetched
git lfs pull

# Run the default build command
vercel
