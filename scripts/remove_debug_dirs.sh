#!/bin/bash
# Script to remove directories ending with '_debug'
find ./bundles -type d -name '*_debug' -exec rm -rf {} +

