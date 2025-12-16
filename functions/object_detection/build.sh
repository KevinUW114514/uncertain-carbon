#!/bin/sh
# set -eu

# 1) Remove any previous OpenCV artifacts from the target dir
# rm -rf "${SRC_PKG}/cv2" \
#        "${SRC_PKG}/opencv_python"* \
#        "${SRC_PKG}"/*opencv*dist-info* \
#        "${SRC_PKG}"/cv2* \
#        "${SRC_PKG}"/_cv2*

# 2) Install dependencies into target dir
# pip3 install --no-cache-dir -r "${SRC_PKG}/requirements.txt" -t "${SRC_PKG}"

# pip3 install --no-cache-dir -r "${SRC_PKG}/requirements-torch.txt" -t "${SRC_PKG}"
# pip3 install --no-cache-dir -r "${SRC_PKG}/requirements-base.txt" -t "${SRC_PKG}"
# pip3 install --no-cache-dir --no-deps ultralytics -t "${SRC_PKG}"
pip3 install -r "${SRC_PKG}/requirements-torch.txt" -t "${SRC_PKG}"
pip3 install -r "${SRC_PKG}/requirements-base.txt" -t "${SRC_PKG}"
pip3 install --no-deps ultralytics -t "${SRC_PKG}"

# 4) Copy built package to deploy package
cp -r "${SRC_PKG}" "${DEPLOY_PKG}"
