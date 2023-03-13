import csv
import re
import time
from abc import abstractmethod
from enum import Enum
from statistics import stdev

import av
import cv2
import gpustat
import numpy as np
import psutil
import torch
from numpy import percentile
from torchvision.transforms.functional import pil_to_tensor
from utils import IterationResult, avg, b_to_mb, plot_list_to_image, s_to_ms

import PyNvCodec as nvc
import PytorchNvCodec as pnvc


class _Tool:
    def __init__(self, file_to_decode: str, with_plot=False, process_name="python3", to_tensor: bool = False) -> None:
        self.file_to_decode = file_to_decode
        self.with_plot = with_plot
        self.records: IterationResult = {
            "fpt": [],
            "cpu": [],
            "mem": [],
            "gpu": [],
            "gpu_mem": []
        }

        self.process_pid = 0
        self.process_name = re.compile(process_name)
        for p in psutil.process_iter(['pid', 'name', 'memory_info']):
            if not self.process_name.match(p.name()):
                continue
            self.process_pid = p.pid

        self._psutil_handle = psutil.Process(self.process_pid)
        self.to_tensor = to_tensor

    @abstractmethod
    def convert_tensor(self, output) -> torch.Tensor:
        """Convert to tensor"""

    def plot_all_metrics_to_png(self, file_name: str):
        for key, value in self.records.items():
            plot_list_to_image(
                value, 'benchmark-results/plot/{}/{}.png'.format(key, file_name))

    # Dump records to CSV files
    def dump_all_records_to_csv(self, file_name: str):
        for key, value in self.records.items():
            with open("benchmark-results/csv/{}/{}.csv".format(key, file_name), 'w') as file_csv:
                writer = csv.writer(file_csv)
                writer.writerow(value)

    def summarize_records(self):
        dict_result = {}

        for key, value in self.records.items():
            sorted_value = sorted(value)
            dict_result[key] = {
                "avg": round(avg(sorted_value), 2) if len(value) != 0 else 0,
                "min": round(min(sorted_value), 2) if len(value) != 0 else 0,
                "max": round(max(sorted_value), 2) if len(value) != 0 else 0,
                "q1": round(percentile(sorted_value, 25), 2) if len(value) != 0 else 0,
                "q2": round(percentile(sorted_value, 50), 2) if len(value) != 0 else 0,
                "q3": round(percentile(sorted_value, 75), 2) if len(value) != 0 else 0,
                "stdev": round(stdev(sorted_value), 2) if len(value) != 0 else 0,
            }

        return dict_result

    def summary_to_csv(self, file_name: str, dict_summary):
        with open('benchmark-results/individual_summary/{}.csv'.format(file_name), 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for _, value in dict_summary.items():
                csv_writer.writerow([*value.values()])

    @abstractmethod
    def decode(self):
        pass


class PyAV(_Tool):
    def __init__(self, file_to_decode: str, with_plot=False, process_name="python3") -> None:
        super().__init__(file_to_decode, with_plot, process_name)

    def convert_tensor(self, output) -> torch.Tensor:
        return pil_to_tensor(output).to("cuda:0")        

    def decode(self, warmup_iteration=0):
        av_input = av.open(self.file_to_decode)
        iteration_count = 1
        for packet in av_input.demux():
            if packet.size == 0:
                continue

            gpu = gpustat.core.GPUStatCollection.new_query()
            start_counter = time.perf_counter()
            (frame,) = packet.decode()
            if self.to_tensor:
                _ = self.convert_tensor(frame)
            end_counter = time.perf_counter()
            gpu_processes = list(filter(lambda x: self.process_name.match(
                x['command']), gpu[0].processes))

            processing_time = round(s_to_ms(end_counter - start_counter), 2)
            cpu_util = psutil.cpu_percent()
            mem_util = round(b_to_mb(self._psutil_handle.memory_info().rss), 2)
            gpu_util = gpu[0].utilization

            if iteration_count > warmup_iteration:
                if len(gpu_processes) > 0:
                    for process in gpu_processes:
                        self.records["gpu_mem"].append(
                            process["gpu_memory_usage"])
                else:
                    self.records["gpu_mem"].append(0)

                self.records["fpt"].append(processing_time)
                self.records["cpu"].append(cpu_util)
                self.records["mem"].append(mem_util)
                self.records["gpu"].append(gpu_util)

            iteration_count += 1

        self.dump_all_records_to_csv(file_name="pyav")

        if self.with_plot:
            self.plot_all_metrics_to_png(file_name='pyav')


class OpenCV(_Tool):
    def __init__(self, file_to_decode: str, with_plot=False, process_name="python3") -> None:
        super().__init__(file_to_decode, with_plot, process_name)

    def convert_tensor(self, output) -> torch.Tensor:
        return torch.from_numpy(output).to("cuda:0")

    def decode(self, warmup_iteration=0):
        video = cv2.VideoCapture(self.file_to_decode)
        iteration_count = 1
        while video.isOpened():
            gpu = gpustat.core.GPUStatCollection.new_query()
            gpu_processes = list(filter(lambda x: self.process_name.match(
                x['command']), gpu[0].processes))

            start_counter = time.perf_counter()
            ret, frame = video.read()
            if not ret:
                break
            if self.to_tensor:
                _ = self.convert_tensor(frame)
            end_counter = time.perf_counter()

            processing_time = round(s_to_ms(end_counter - start_counter), 2)
            cpu_util = psutil.cpu_percent()
            mem_util = round(b_to_mb(self._psutil_handle.memory_info().rss), 2)
            gpu_util = gpu[0].utilization

            if iteration_count > warmup_iteration:
                if len(gpu_processes) > 0:
                    for process in gpu_processes:
                        self.records["gpu_mem"].append(
                            process["gpu_memory_usage"])
                else:
                    self.records["gpu_mem"].append(0)

                self.records["fpt"].append(processing_time)
                self.records["cpu"].append(cpu_util)
                self.records["mem"].append(mem_util)
                self.records["gpu"].append(gpu_util)

            iteration_count += 1

        video.release()

        self.dump_all_records_to_csv(file_name="opencv")

        if self.with_plot:
            self.plot_all_metrics_to_png(file_name='opencv')


class DecodeStatus(Enum):
    # Decoding error.
    DEC_ERR = 0,
    # Frame was submitted to decoder.
    # No frames are ready for display yet.
    DEC_SUBM = 1,
    # Frame was submitted to decoder.
    # There's a frame ready for display.
    DEC_READY = 2


class NVDec(_Tool):
    def __init__(self, file_to_decode: str, with_plot=False, process_name="python3") -> None:
        super().__init__(file_to_decode, with_plot, process_name)
        self.gpu_id = 0
        self.nv_dec = nvc.PyNvDecoder(file_to_decode, self.gpu_id)
        if self.to_tensor:
            self.to_yuv = nvc.PySurfaceConverter(self.nv_dec.Width(), self.nv_dec.Height(), self.nv_dec.PixelFormat.NV12, self.nv_dec.PixelFormat.YUV420, self.gpu_id)
            self.to_rgb = nvc.PySurfaceConverter(self.nv_dec.Width(), self.nv_dec.Height(), self.nv_dec.PixelFormat.YUV420, self.nv_dec.PixelFormat.RGB, self.gpu_id)
            self.to_pln = nvc.PySurfaceConverter(self.nv_dec.Width(), self.nv_dec.Height(), self.nv_dec.PixelFormat.RGB, self.nv_dec.PixelFormat.RGB_PLANAR, self.gpu_id)
            self.cc_ctx = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)
        # Numpy array to store decoded frames pixels
        self.frame_nv12 = np.ndarray(shape=(0), dtype=np.uint8)
        # Encoded packet data
        self.packet_data = nvc.PacketData()

    def convert_tensor(self, output) -> torch.Tensor:
        # output should be rgb24_planar
        surf_plane = output.PlanePtr()
        img_tensor = pnvc.makefromDevicePtrUint8(surf_plane.GpuMem(),
                                                 surf_plane.Width(),
                                                 surf_plane.Height(),
                                                 surf_plane.Pitch(),
                                                 surf_plane.ElemSize())
        return img_tensor        

    def decode_frame_tensor(self) -> DecodeStatus:
        status = DecodeStatus.DEC_ERR
        try:
            nv12_surface = self.nv_dec.DecodeSingleSurface()
            if nv12_surface.Empty():
                print('Can not decode frame')
                return status

            # Convert from NV12 to YUV420.
            # This extra step is required because not all NV12 -> RGB conversions
            # implemented in NPP support all color spaces and ranges.
            yuv420 = self.to_yuv.Execute(nv12_surface, self.cc_ctx)
            if yuv420.Empty():
                print('Can not convert nv12 -> yuv420')
                return status

            # Convert from YUV420 to interleaved RGB.
            rgb24_small = self.to_rgb.Execute(yuv420, self.cc_ctx)
            if rgb24_small.Empty():
                print('Can not convert yuv420 -> rgb')
                return status

            # Convert to planar RGB.
            rgb24_planar = self.to_pln.Execute(rgb24_small, self.cc_ctx)
            if rgb24_planar.Empty():
                print('Can not convert rgb -> rgb planar')
                return status
            _ = self.convert_tensor(rgb24_planar)
            status = DecodeStatus.DEC_READY
        except Exception as e:
            print(getattr(e, 'message', str(e)))
        return status

    # Decode single video frame
    def decode_frame(self) -> DecodeStatus:
        status = DecodeStatus.DEC_ERR

        try:
            frame_ready = False
            frame_ready = self.nv_dec.DecodeSingleFrame(
                self.frame_nv12, self.packet_data)

            # Nvdec is sync in this mode so if frame isn't returned it means
            # EOF or error.
            if frame_ready:
                status = DecodeStatus.DEC_READY
            else:
                return status

        except Exception as e:
            print(getattr(e, 'message', str(e)))

        return status

    # Decode all available video frames and write them to output file.
    def decode(self, warmup_iteration=0) -> IterationResult:
        iteration_count = 1

        # Main decoding cycle
        while True:

            start_counter = time.perf_counter()
            if not self.to_tensor:
                status = self.decode_frame()
            else:
                status - self.decord_frame_tensor()
            if status == DecodeStatus.DEC_ERR:
                break
            end_counter = time.perf_counter()
            gpu = gpustat.core.GPUStatCollection.new_query()
            gpu_processes = list(filter(lambda x: self.process_name.match(
                x['command']), gpu[0].processes))

            processing_time = round(s_to_ms(end_counter - start_counter), 2)
            cpu_util = psutil.cpu_percent()
            mem_util = round(b_to_mb(self._psutil_handle.memory_info().rss), 2)
            gpu_util = gpu[0].utilization

            if iteration_count > warmup_iteration:
                if len(gpu_processes) > 0:
                    for process in gpu_processes:
                        self.records["gpu_mem"].append(
                            process["gpu_memory_usage"])
                else:
                    self.records["gpu_mem"].append(0)

                self.records["fpt"].append(processing_time)
                self.records["cpu"].append(cpu_util)
                self.records["mem"].append(mem_util)
                self.records["gpu"].append(gpu_util)

            iteration_count += 1

        self.dump_all_records_to_csv(file_name="nvdec")

        if self.with_plot:
            self.plot_all_metrics_to_png(file_name='nvdec')
