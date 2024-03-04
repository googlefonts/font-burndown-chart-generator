"""
This type stub file was generated by pyright.
"""

from matplotlib import backend_bases
from matplotlib.backends import backend_agg
from matplotlib.backend_bases import _Backend

"""
Displays Agg images in the browser, with interactivity
"""
_log = ...
_SPECIAL_KEYS_LUT = ...
class TimerTornado(backend_bases.TimerBase):
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class TimerAsyncio(backend_bases.TimerBase):
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class FigureCanvasWebAggCore(backend_agg.FigureCanvasAgg):
    manager_class = ...
    _timer_cls = TimerAsyncio
    supports_blit = ...
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def show(self): # -> None:
        ...
    
    def draw(self): # -> None:
        ...
    
    def blit(self, bbox=...): # -> None:
        ...
    
    def draw_idle(self): # -> None:
        ...
    
    def set_cursor(self, cursor): # -> None:
        ...
    
    def set_image_mode(self, mode): # -> None:
        """
        Set the image mode for any subsequent images which will be sent
        to the clients. The modes may currently be either 'full' or 'diff'.

        Note: diff images may not contain transparency, therefore upon
        draw this mode may be changed if the resulting image has any
        transparent component.
        """
        ...
    
    def get_diff_image(self): # -> bytes | None:
        ...
    
    def handle_event(self, event): # -> Any | None:
        ...
    
    def handle_unknown_event(self, event): # -> None:
        ...
    
    def handle_ack(self, event): # -> None:
        ...
    
    def handle_draw(self, event): # -> None:
        ...
    
    handle_scroll = ...
    handle_key_release = ...
    def handle_toolbar_button(self, event): # -> None:
        ...
    
    def handle_refresh(self, event): # -> None:
        ...
    
    def handle_resize(self, event): # -> None:
        ...
    
    def handle_send_image_mode(self, event): # -> None:
        ...
    
    def handle_set_device_pixel_ratio(self, event): # -> None:
        ...
    
    def handle_set_dpi_ratio(self, event): # -> None:
        ...
    
    def send_event(self, event_type, **kwargs): # -> None:
        ...
    


_ALLOWED_TOOL_ITEMS = ...
class NavigationToolbar2WebAgg(backend_bases.NavigationToolbar2):
    toolitems = ...
    def __init__(self, canvas) -> None:
        ...
    
    def set_message(self, message): # -> None:
        ...
    
    def draw_rubberband(self, event, x0, y0, x1, y1): # -> None:
        ...
    
    def remove_rubberband(self): # -> None:
        ...
    
    def save_figure(self, *args): # -> None:
        """Save the current figure"""
        ...
    
    def pan(self): # -> None:
        ...
    
    def zoom(self): # -> None:
        ...
    
    def set_history_buttons(self): # -> None:
        ...
    


class FigureManagerWebAgg(backend_bases.FigureManagerBase):
    _toolbar2_class = ...
    ToolbarCls = NavigationToolbar2WebAgg
    def __init__(self, canvas, num) -> None:
        ...
    
    def show(self): # -> None:
        ...
    
    def resize(self, w, h, forward=...): # -> None:
        ...
    
    def set_window_title(self, title): # -> None:
        ...
    
    def add_web_socket(self, web_socket): # -> None:
        ...
    
    def remove_web_socket(self, web_socket): # -> None:
        ...
    
    def handle_json(self, content): # -> None:
        ...
    
    def refresh_all(self): # -> None:
        ...
    
    @classmethod
    def get_javascript(cls, stream=...): # -> str | None:
        ...
    
    @classmethod
    def get_static_file_path(cls): # -> str:
        ...
    


@_Backend.export
class _BackendWebAggCoreAgg(_Backend):
    FigureCanvas = FigureCanvasWebAggCore
    FigureManager = FigureManagerWebAgg

