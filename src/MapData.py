import re
from tkinter import filedialog
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
from random import randint
import json
import struct
import matplotlib.pyplot as plt
from typing import Callable, List, Union
import pywt
from scipy.interpolate import interp1d


class MapData:
    """Class for handling hyperspectral images stored in the .libsdata file format
    """

    def __init__(
        self,
        file_path: str = None,
        overwrite: bool = False
    ) -> None:
        if file_path is None:
            self.file_path = Path(
                filedialog.askopenfilename(
                    filetypes=[('LIBS data', '*.libsdata')]
                )
            )
        else:
            self.file_path = Path(file_path)

        self.BYTE_SIZE = 4
        self.data_type = None
        self.metadata = None
        self.random_spectra_from_batches = None
        self.random_spectrum = None
        self.baselines = None
        self.spectra = None
        self.wvl = None
        self.map_dimensions = None
        self.line_intensities = None
        self.overwrite = overwrite
        self.systemic_noise_spectrum = None

    def get_map_dimensions(self) -> None:
        """Gets the measured map's dimensions (in pixels) assuming that the filename contains this information
        """
        print('getting map dimensions')
        map_dimensions = re.findall(
            '[0-9]{3}x[0-9]{3}',
            self.file_path.name
        )[0].split('x')

        self.map_dimensions = [int(x) for x in map_dimensions]

    def get_metadata(self) -> None:
        """Load metadata from the metadata file corresponding to the selected data file
        """
        print('loading metadata')
        metadata_path = self.file_path.with_suffix('.libsmetadata')
        if metadata_path.is_file():
            with open(
                self.file_path.with_suffix('.libsmetadata'), 'r'
            ) as file:
                self.metadata = json.load(file)
        else:
            raise ImportError('Metadata file is missing')

    def create_data_type(self) -> None:
        """Defines the data_type used for loading in the binary data (takes information from the metadata)
        """
        if self.metadata is None:
            self.get_metadata()
        self.data_type = np.dtype(
            [(
                'data',
                np.float32,
                self.metadata.get('wavelengths')
            )]
        )

    def load_wavelenths(self) -> None:
        """Load the wavelength vector from the binary data file
        """
        print('loading wavelengths')
        if self.data_type is None:
            self.create_data_type()
        self.wvl = np.fromfile(
            self.file_path,
            self.data_type,
            count=1
        )['data'][0]

    def load_batch_of_spectra(
        self,
        batch_size: int,
        start_ndx: int
    ) -> None:
        """Load a batch of consecutive spectra from the binary data file

        Args:
            batch_size (int): number of spectra to load
            start_ndx (int): index of the first spectrum in the batch (in the whole data file)
        """
        if self.data_type is None:
            self.create_data_type()
        if batch_size + start_ndx + 1 < self.metadata.get('spectra'):
            self.batch_spectra = np.fromfile(
                self.file_path,
                self.data_type,
                count=batch_size,
                offset=(1+start_ndx) * self.metadata.get('wavelengths') *
                self.BYTE_SIZE  # 1 for skipping wavelengths
            )['data']
        else:
            print('The chosen batchsize and offset are out of bounds')

    def load_random_spectrum_from_batch(
        self,
        batch_size: int
    ) -> None:
        """Loads a single spectrum from every batch defined by the batch_size parameter

        Args:
            batch_size (int): Number of spectra from which 1 is randomly sampled
        """
        batch_count = self.metadata.get('spectra') // batch_size
        debug_log.debug(batch_count)

        data = []
        with open(self.file_path, 'rb') as source:
            source.seek(
                self.metadata.get('wavelengths') * self.BYTE_SIZE,
                0
            )

            for _ in range(batch_count):
                ndx = randint(0, batch_size)
                debug_log.debug(f'{ndx}')

                source.seek(
                    self.metadata.get('wavelengths') *
                    self.BYTE_SIZE * (ndx - 1),
                    1
                )

                for _ in range(self.metadata.get('wavelengths')):
                    data.extend(
                        struct.unpack(
                            'f',
                            source.read(self.BYTE_SIZE)
                        )
                    )

                if ndx != batch_size:
                    source.seek(
                        self.metadata.get('wavelengths') *
                        self.BYTE_SIZE * (batch_size - ndx - 1),
                        1
                    )

        self.random_spectra_from_batches = np.reshape(
            data,
            (-1, self.metadata.get('wavelengths'))
        )

    def load_random_spectrum(self) -> None:
        """Load a random spectrum from the whole data file
        """
        if self.data_type is None:
            self.create_data_type()
        chosen_ndx = randint(1, self.metadata.get('spectra'))
        self.random_spectrum = np.fromfile(
            self.file_path,
            self.data_type,
            count=1,
            offset=chosen_ndx *
            self.metadata.get('wavelengths') * self.BYTE_SIZE
        )['data'][0]

    def plot_random_spectrum(
        self,
        return_fig: bool = False
    ) -> None:
        """load and plot a random spectrum from the file
        """
        fig, ax = plt.subplots()
        if not hasattr(self, 'wvl'):
            self.load_wavelenths()

        self.load_random_spectrum()

        ax.plot(
            self.wvl,
            self.random_spectrum
        )
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity (counts)')
        fig.show()

        if return_fig:
            return fig
        else:
            return None

    def load_all_data(
        self,
        file_name_supplement: str
    ) -> None:
        """loads all spectra from the file
        """
        if not self._check_file(file_name_supplement):
            print('preprocessed file was not found; setting overwrite to True')
            self.overwrite = True

        if self.overwrite:
            print(
                'loading raw data'
            )
            if self.data_type is None:
                self.create_data_type()
            self.spectra = np.fromfile(
                self.file_path,
                self.data_type,
                offset=self.metadata.get('wavelengths') * self.BYTE_SIZE
            )['data']
        elif self._check_file(file_name_supplement) and not self.overwrite:
            self.spectra = np.load(
                self.file_path.with_name(
                    self._supplement_file_name(file_name_supplement)
                ),
                allow_pickle=False
            )

    def trim_spectra(
        self,
        trim_width: int
    ) -> None:
        """Removes the edges of the spectra. To be used if the intensity abruptly drops towards the ends.

        Args:
            trim_width (int): The number of pixels to drop at both the beginning and end of every spectrum.
        """
        if self.overwrite:
            self.spectra = self.spectra[:, trim_width:-trim_width]
        self.wvl = self.wvl[trim_width:-trim_width]

    @staticmethod
    def _rolling_min(
        arr: np.array,
        window_width: int
    ) -> np.array:
        """Calculates the moving minima in each row of the provided array.

        Args:
            arr (np.array): A 2D array with each row representing a spectrum.
            window_width (int): The width of the window where the minimum is to be found.

        Returns:
            np.array: A 2D array of the moving minima.
        """
        window = sliding_window_view(
            arr,
            (window_width,),
            axis=len(arr.shape) - 1
        )

        return np.amin(window, axis=len(arr.shape))

    @staticmethod
    def _get_smoothing_kernel(window_width: int) -> np.array:
        """Generates a Gaussian smoothin kernel of the desired width.

        Args:
            window_width (int): Width of the kernel (length of the resulting vector).

        Returns:
            np.array: A Gaussian distribution with it's center at the vectors middle.
        """
        kernel = np.arange(-window_width//2, window_width//2 + 1, 1)
        sigma = window_width // 4
        kernel = np.exp(-(kernel ** 2) / (2 * sigma**2))
        kernel /= kernel.sum()

        return kernel

    def get_baseline(
        self,
        min_window_size: int = 50,
        smooth_window_size: int = None
    ) -> None:
        """
        Determines the spectra's baselines.

        Args:
            min_window_size (int, optional): Width of the rolling minimum function. Defaults to 50.
            smooth_window_size (int, optional): Width of the smoothing function. Defaults to None.
        """
        if self.overwrite and self.baselines is None:
            print('getting baselines')
            if smooth_window_size is None:
                smooth_window_size = 2*min_window_size
            local_minima = self._rolling_min(
                arr=np.hstack(
                    [self.spectra[:, 0][:, np.newaxis]] *
                    ((min_window_size + smooth_window_size) // 2)
                    + [self.spectra]
                    + [self.spectra[:, -1][:, np.newaxis]] *
                    ((min_window_size + smooth_window_size) // 2)
                ),
                window_width=min_window_size
            )
            self.baselines = np.apply_along_axis(
                arr=local_minima,
                func1d=np.convolve,
                axis=1,
                v=self._get_smoothing_kernel(smooth_window_size),
                mode='valid'
            )

    def _align_baselines_with_spectra(self) -> None:
        """
        Discards the last few pixels of the determined baselines if they are longer than the corresponding spectra.
        """
        self.baselines = self.baselines[
            :,
            :-(self.baselines.shape[1] - self.spectra.shape[1])
        ]

    def baseline_correct(
        self,
        keep_baselines: bool = False
    ) -> None:
        """
        Subtracts the baselines from the spectra.

        Args:
            keep_baselines (bool, optional): Whether to keep or discard the baselines. Defaults to False.
        """
        if self.overwrite:
            self.get_baseline()
            self._align_baselines_with_spectra()
            self.spectra = np.subtract(
                self.spectra,
                self.baselines
            )

            if not keep_baselines:
                self.baselines = None

    def get_emission_line_intensities(
        self,
        left_boundaries: list,
        right_boundaries: list,
        line_centers: list,
        intensity_funcs: List[Callable]
    ) -> None:
        """_summary_

        Args:
            left_boundaries (list): _description_
            right_boundaries (list): _description_
            line_centers (list): _description_
            intensity_func (Callable): _description_

        Raises:
            ValueError: _description_
        """
        if len(left_boundaries) != len(right_boundaries) != len(line_centers):
            raise ValueError('incompatible lists provided')

        self.line_intensities = dict()
        for intensity_func in intensity_funcs:
            print(
                f'extracting emission line intensities using {intensity_func.__name__}')
            self.line_intensities[intensity_func.__name__] = dict()
            for line_center, left_bound, right_bound in zip(
                line_centers,
                left_boundaries,
                right_boundaries
            ):
                self.line_intensities[intensity_func.__name__][
                    f'{self.wvl[line_center]:.2f}'
                ] = intensity_func(
                    self.spectra[:, left_bound:right_bound],
                    axis=1
                )

    def vector_to_array(
        self,
        data: np.array
    ) -> None:
        """_summary_

        Args:
            data (np.array): _description_

        Returns:
            _type_: _description_
        """
        data = data.copy().reshape(self.map_dimensions[::-1])
        data[::2, :] = data[::2, ::-1]

        return data

    @staticmethod
    def _upsample_wvl(wvl: np.array) -> np.array:
        return np.linspace(
            start=wvl[0],
            stop=wvl[-1],
            num=int(2 ** np.ceil(np.log2(len(wvl))))
        )

    @staticmethod
    def _upsample_spectrum(
        spectrum: np.array,
        wvl: np.array,
        new_wvl: np.array = None
    ) -> np.array:
        """
        Interpolates a spectrum.

        Args:
            spectrum (np.array): The spectrum to interpolate.
            wvl (np.array): The initial wavelength vector.
            new_wvl (np.array, optional): The new wavelength vector. Defaults to None.

        Returns:
            np.array: Interpolated spectrum.
        """
        return interp1d(wvl, spectrum)(new_wvl)

    def upsample_spectra(self) -> None:
        """
        Upsamples the spectra.
        """
        if self.overwrite:
            print('upsampling spectra')
            self.spectra = np.apply_along_axis(
                arr=self.spectra,
                axis=1,
                func1d=self._upsample_spectrum,
                wvl=self.wvl,
                new_wvl=self._upsample_wvl(self.wvl)
            )
        self.wvl = self._upsample_wvl(self.wvl)

    @staticmethod
    def _denoise_spectrum(
        spectrum: np.array,
        wavelet: pywt.Wavelet,
        threshold: Union[float, Callable],
        level: int = 9
    ) -> np.array:
        """
        Denoise a given spectrum using the provided wavelet, threshold value, and level of decomposition.
        TODO test the removed noise's distribution for normality

        Args:
            spectrum (np.array): The spectrum to be denoised.
            wavelet (pywt.Wavelet): The wavelet used for decomposition.
            threshold (Union[float, Callable]): The threshold value or function used to threshold the coefficients.
            level (int): The depth of decomposition.

        Returns:
            np.array: The denoised spectrum.
        """
        wavelet_docomposition = pywt.swt(
            spectrum,
            wavelet=wavelet,
            level=level,
            start_level=0,
            trim_approx=False
        )

        if isinstance(threshold, Callable):
            threshold = threshold(spectrum)

        thresholded_decomposition = [
            (
                pywt.threshold(
                    data=coefs[0],
                    substitute=0,
                    value=threshold,
                    mode='soft'
                ),
                pywt.threshold(
                    data=coefs[1],
                    substitute=0,
                    value=threshold,
                    mode='soft'
                )
            )
            for coefs
            in wavelet_docomposition
        ]

        return pywt.iswt(
            thresholded_decomposition,
            wavelet=wavelet
        )

    def estimate_systemic_noise(self) -> None:
        """
        Estimate the systemic noise spectrum.

        Returns:
            None
        """
        if self.overwrite:
            print('estimating systemic noise spectrum')
            diff_spectra = np.diff(self.spectra[:, :])
            self.systemic_noise_spectrum = np.median(
                diff_spectra,
                axis=0,
                keepdims=True
            ) / 2

    def denoise_spectra(
        self,
        file_name_supplement: str,
        wavelet: pywt.Wavelet = pywt.Wavelet('rbio6.8'),
        threshold: Union[float, Callable] = 35.,
        level: int = 9
    ) -> None:
        """
        Apply wavelet denoising to the spectra data along the second axis.

        Args:
            file_name_supplement (str): A string to append to the file name stem.
            wavelet (pywt.Wavelet): wavelet to use for the transformation (default: 'rbio6.8')
            threshold (threshold: float or callable): threshold for wavelet coefficients (default: 35.)
            level (int): wavelet decomposition level (default: 9)

        Returns:
            None
        """
        if self.overwrite:
            print('denoising spectra')
            self.spectra = np.apply_along_axis(
                func1d=self._denoise_spectrum,
                axis=1,
                arr=self.spectra,
                wavelet=wavelet,
                threshold=threshold,
                level=level
            )
            self.save_spectra(file_name_supplement)

    def _supplement_file_name(
        self,
        file_name_supplement: str
    ) -> str:
        """
        Concatenates a given string to the file name stem, and returns the updated file
        name with the .npy extension.

        Args:
            file_name_supplement (str): A string to append to the file name stem.

        Returns:
            str: Updated file name with the .npy extension.
        """
        return f'{self.file_path.stem}_{file_name_supplement}.npy'

    def _check_file(
        self,
        file_name_supplement: str
    ) -> bool:
        """Checks if a file exists in the same directory with a given file name supplement.

        Args:
            file_name_supplement (str): A string representing the file name supplement to add to the file name.

        Returns:
            bool: A boolean indicating if the file exists.
        """
        return self.file_path.with_name(
            self._supplement_file_name(file_name_supplement)
        ).exists()

    def save_spectra(
        self,
        file_name_supplement: str
    ) -> None:
        """
        Save the current array of spectra to disk.

        Args:
            file_name_supplement (str): A supplement to the filename to help differentiate the saved file.
        """
        print('saving spectra')
        np.save(
            arr=self.spectra,
            file=self.file_path.with_name(
                self._supplement_file_name(file_name_supplement)
            )
        )


def min_max_dist(
    arr: np.array,
    axis: int = 1
) -> np.array:
    """
    Calculates the range between the maximum and minimum values of an array along a specified axis.

    Args:
        arr (np.array): The input array.
        axis (int): The axis along which to calculate the range. Default is 1.

    Returns:
        np.array: The range of the array along the specified axis.
    """
    return np.max(arr, axis=axis) - np.min(arr, axis=axis)


def get_triangular_kernel(size: int) -> np.array:
    """
    Generates a triangular kernel of a given size.

    Args:
        size (int): an integer representing the size of the kernel.

    Returns:
        np.array: na array representing the triangular kernel.
    """
    return np.concatenate((np.arange(1, size), np.arange(size, 0, -1))) / size


def triangle_corr(
    arr: np.array,
    axis: int = 1
) -> np.array:
    """
    Calculates the correlation coefficient of an array with a triangular kernel along a specified axis.

    Args:
        arr (np.array): The input array.
        axis (int, optional):  The axis along which to calculate the correlation. Defaults to 1.

    Returns:
        np.array: The correlation coefficient of the array with the triangular kernel along the specified axis.
    """
    size = np.ceil(arr.shape[1] / 2)
    kernel = get_triangular_kernel(int(size))
    kernel = np.interp(
        np.linspace(0, len(kernel), num=arr.shape[1]),
        np.arange(len(kernel)),
        get_triangular_kernel(size)
    )

    return np.apply_along_axis(
        func1d=lambda row: np.corrcoef(row, kernel)[0, 1],
        arr=arr,
        axis=axis
    )
