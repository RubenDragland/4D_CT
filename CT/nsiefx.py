# Copyright (c) 2022 North Star Imaging, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#   * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#     disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#     following disclaimer in the documentation and/or other materials provided with the distribution.
#   * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
#     products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from ctypes import *
from enum import IntEnum

shared_lib_path = "./efX-SDK.dll"
try:
    efx_sdk = CDLL(shared_lib_path)
    print("Successfully loaded ", efx_sdk)
except Exception as e:
    raise e

vol_create = efx_sdk.nsi_efx_volume_create
vol_create.restype = c_void_p

vol_delete = efx_sdk.nsi_efx_volume_delete
vol_delete.argtypes = [c_void_p]

vol_open = efx_sdk.nsi_efx_volume_wopen
vol_open.restype = c_uint
vol_open.argtypes = [c_void_p, c_wchar_p]

vol_close = efx_sdk.nsi_efx_volume_close
vol_close.restype = c_bool
vol_close.argtypes = [c_void_p]

vol_slice_width = efx_sdk.nsi_efx_volume_slice_width
vol_slice_width.argtypes = [c_void_p, c_void_p]

vol_slice_height = efx_sdk.nsi_efx_volume_slice_height
vol_slice_height.argtypes = [c_void_p, c_void_p]

vol_num_slices = efx_sdk.nsi_efx_volume_num_slices
vol_num_slices.argtypes = [c_void_p, c_void_p]

vol_vmin = efx_sdk.nsi_efx_volume_vmin
vol_vmin.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p]

vol_vmax = efx_sdk.nsi_efx_volume_vmax
vol_vmax.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p]

vol_voxel_size = efx_sdk.nsi_efx_volume_voxel_size
vol_voxel_size.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p]

vol_data_min = efx_sdk.nsi_efx_volume_data_min
vol_data_min.argtypes = [c_void_p, c_void_p]

vol_data_max = efx_sdk.nsi_efx_volume_data_max
vol_data_max.argtypes = [c_void_p, c_void_p]

vol_read_slice = efx_sdk.nsi_efx_volume_read_slice
vol_read_slice.restype = c_bool
vol_read_slice.argtypes = [c_void_p, c_void_p, c_uint]

vol_sdk_version = efx_sdk.nsi_efx_volume_sdk_version
vol_sdk_version.restype = c_uint
vol_sdk_version.argtypes = [c_void_p, c_char_p, c_uint]

save_gray_tif32 = efx_sdk.nsi_efx_save_gray_tif32_w
save_gray_tif32.restype = c_bool
save_gray_tif32.argtypes = [c_wchar_p, c_void_p, c_uint, c_uint]


class NsiError(IntEnum):
    SUCCESS = 0
    ERROR_INVALID_FILE = 100
    ERROR_UNSUPPORTED_HDR_VERSION = 200
    ERROR_UNSPECIFIED = 9999


class efXVolume:
    def __init__(self, handle):
        self.handle = handle

    def open(self, file):
        return vol_open(self.handle, file)

    def close(self):
        return vol_close(self.handle)

    def slice_width(self):
        width = c_uint32()
        vol_slice_width(self.handle, byref(width))
        return width.value

    def slice_height(self):
        height = c_uint32()
        vol_slice_height(self.handle, byref(height))
        return height.value

    def num_slices(self):
        num = c_uint32()
        vol_num_slices(self.handle, byref(num))
        return num.value

    def vmin(self):
        x = c_double()
        y = c_double()
        z = c_double()
        vol_vmin(self.handle, byref(x), byref(y), byref(z))
        return x.value, y.value, z.value

    def vmax(self):
        x = c_double()
        y = c_double()
        z = c_double()
        vol_vmax(self.handle, byref(x), byref(y), byref(z))
        return x.value, y.value, z.value

    def voxel_size(self):
        x = c_double()
        y = c_double()
        z = c_double()
        vol_voxel_size(self.handle, byref(x), byref(y), byref(z))
        return x.value, y.value, z.value

    def data_min(self):
        min = c_double()
        vol_data_min(self.handle, byref(min))
        return min.value

    def data_max(self):
        max = c_double()
        vol_data_max(self.handle, byref(max))
        return max.value

    def read_slice(self, sliceidx):
        slice = ((c_float * self.slice_width()) * self.slice_height())()
        if not vol_read_slice(
            self.handle, cast(slice, c_void_p), c_uint(int(sliceidx))
        ):
            raise Exception("Failed to read slice")
        return slice

    def sdk_version(self):
        buffer_length = 255
        buffer = create_string_buffer(buffer_length)
        vol_sdk_version(self.handle, buffer, buffer_length)
        return buffer.value.decode()


def open(filename):
    class VolAccessor:
        def __init__(self, file):
            self.filename = file

        def __enter__(self):
            self.handle = vol_create()
            self.volume = efXVolume(self.handle)
            try:
                err = self.volume.open(self.filename)
                if err != NsiError.SUCCESS:
                    raise Exception("Failed to open file", err)
            except Exception:
                vol_delete(self.handle)
                raise
            return self.volume

        def __exit__(self, exc_type, exc_val, exc_tb):
            vol_delete(self.handle)

    return VolAccessor(filename)


def save_tif32(filename, slice, slice_height, slice_width):
    if isinstance(slice, list):
        cslice = (c_float * len(slice))(*slice)
    else:
        cslice = slice
    if not save_gray_tif32(
        filename, cast(cslice, c_void_p), c_uint(slice_height), c_uint(slice_width)
    ):
        raise Exception("Failed to save tif")
