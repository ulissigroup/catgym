import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
from ase.io.utils import PlottingVariables, make_patch_list
from ase.io.eps import EPS

class PNG(EPS):
    def write_header(self):
        from matplotlib.backends.backend_agg import RendererAgg

        try:
            from matplotlib.transforms import Value
        except ImportError:
            dpi = 72
        else:
            dpi = Value(72)

        self.renderer = RendererAgg(self.w, self.h, dpi)

    def write(self):
        self.write_header()
        self.write_body()
        renderer = self.renderer
        x = renderer.buffer_rgba()
        img_array = np.frombuffer(x, np.uint8).reshape(
                        (int(self.h), int(self.w), 4))
        return img_array[:,:,:3]

def render_image(atoms, **parameters):
    return PNG(atoms, **parameters).write()


