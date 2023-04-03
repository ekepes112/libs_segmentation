
class MapData:
  """Class for handling hyperspectral images stored in the .libsdata file format
  """
  def __init__(self):
    self.file_path = Path(
      filedialog.askopenfilename(filetypes=[('LIBS data','*.libsdata')])
    )
    self.BYTE_SIZE = 4


  def get_map_dimensions(self):
    """Gets the measured map's dimensions (in pixels) assuming that the filename contains this information
    """
    map_dimensions = re.findall(
        '[0-9]{3}x[0-9]{3}',
        self.file_path.name
    )[0].split('x')

    self.map_dimensions = [int(x) for x in map_dimensions]


  def get_metadata(self):
    """Load metadata from the metadata file corresponding to the selected data file
    """
    metadata_path = self.file_path.with_suffix('.libsmetadata')
    if metadata_path.is_file():
      with open(
        self.file_path.with_suffix('.libsmetadata'),'r'
      ) as file:

        self.metadata = json.load(file)
    else:
      raise ImportError('Metadata file is missing')
    

  def load_wavelenths(self):
    with open(self.file_path,'rb') as source:        
        self.wvl = []
        
        for _ in range(self.metadata.get('wavelengths')):
          self.wvl.extend(
            struct.unpack(
                'f',source.read(self.BYTE_SIZE)
            )
          )

        self.wvl = np.array(self.wvl)


  def load_batch_of_spectra(self, batch_size, start_ndx):
    with open(self.file_path,'rb') as source:
      source.seek(
        self.metadata.get('wavelengths') * self.BYTE_SIZE,
        (1 + start_ndx) * self.metadata.get('wavelengths') * self.BYTE_SIZE
      )

      data = []
      for _ in range(self.metadata.get('wavelengths') * batch_size):

        data.extend(
          struct.unpack(
            'f',source.read(self.BYTE_SIZE)
          )
        )

      self.data = np.reshape(
          data,
          (-1,self.metadata.get('wavelengths'))
      )


  def load_random_spectrum_from_batch(self, batch_size):
    with open(self.file_path,'rb') as source:
      chosen_ndx = randint(0,batch_size)
      source.seek(
        self.metadata.get('wavelengths') * self.BYTE_SIZE * (chosen_ndx + 1),
        0
      )

      data = []
      for _ in range(self.metadata.get('wavelengths')):

        data.extend(
          struct.unpack(
            'f',source.read(self.BYTE_SIZE)
          )
        )

      self.data = np.reshape(
          data,
          (-1,self.metadata.get('wavelengths'))
      )

  
  def load_random_spectrum(self):
    data = []
    with open(self.file_path,'rb') as source:
      chosen_ndx = randint(1,self.metadata.get('spectra'))
      source.seek(
        self.metadata.get('wavelengths') * self.BYTE_SIZE * chosen_ndx,
        0
      )
      for _ in range(self.metadata.get('wavelengths')):
        data.extend(
          struct.unpack(
            'f',source.read(self.BYTE_SIZE)
          )
        )
      return(np.array(data))


  def plot_random_spectrum(self):
    """plot a random spectrum from the file
    """
    fig,ax = plt.subplots()
    if not self.load_wavelenths:
      self.load_wavelenths()

    ax.plot(
      self.wvl,
      self.load_random_spectrum()
    )
    fig.show()
