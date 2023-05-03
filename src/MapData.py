import re
from tkinter import filedialog
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
from random import randint
import json
import struct
import matplotlib.pyplot as plt
from typing import Callable, List


class MapData:
    """Class for handling hyperspectral images stored in the .libsdata file format
    """

    def __init__(self, file_path: str = None):
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

    def get_map_dimensions(self) -> None:
        """Gets the measured map's dimensions (in pixels) assuming that the filename contains this information
        """
        map_dimensions = re.findall(
            '[0-9]{3}x[0-9]{3}',
            self.file_path.name
        )[0].split('x')

        self.map_dimensions = [int(x) for x in map_dimensions]

    def get_metadata(self) -> None:
        """Load metadata from the metadata file corresponding to the selected data file
        """
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

    def load_all_data(self) -> None:
        """loads all spectra from the file
        """
        if self.data_type is None:
            self.create_data_type()
        self.spectra = np.fromfile(
            self.file_path,
            self.data_type,
            offset=self.metadata.get('wavelengths') * self.BYTE_SIZE
        )['data']

    def trim_spectra(
        self,
        trim_width: int
    ) -> None:
        """Removes the edges of the spectra. To be used if the intensity abruptly drops towards the ends.

        Args:
            trim_width (int): The number of pixels to drop at both the beginning and end of every spectrum.
        """
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
        """Determines the spectra's baselines.

        Args:
            min_window_size (int, optional): Width of the rolling minimum function. Defaults to 50.
            smooth_window_size (int, optional): Width of the smoothing function. Defaults to None.
        """
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

    def align_baselines_with_spectra(self) -> None:
        """Discards the last few pixels of the determined baselines if they are longer than the corresponding spectra.
        """
        self.baselines = self.baselines[
            :,
            :-(self.baselines.shape[1] - self.spectra.shape[1])
        ]

    def baseline_correct(
        self,
        keep_baselines: bool = False
    ) -> None:
        """_summary_

        Args:
            keep_baselines (bool, optional): _description_. Defaults to False.
        """
        if self.baselines is None:
            self.get_baseline()

        self.align_baselines_with_spectra()

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
        # intensity_func: Callable
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
