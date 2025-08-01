"""
a tkinter frame
"""
import os
import tkinter as tk
from tkinter import ttk

from ...env_functions import get_dls_visits, MMG_BEAMLINES
from ..misc.logging import create_logger
from ..misc.config import get_config, save_config, C
from ..misc.functions import select_folder, show_error

logger = create_logger(__file__)


class TitleWindow:
    def __init__(self, root: tk.Misc, config: dict | None = None):
        self.root = root
        self.config = get_config() if config is None else config

        self.beamline = tk.StringVar(self.root, '')
        self.visit = tk.StringVar(self.root, '')
        self.data_dir = tk.StringVar(self.root, '')
        self.proc_dir = tk.StringVar(self.root, '')
        self.notebook_dir = tk.StringVar(self.root, '')
        self.visits = {}

        frm = ttk.Frame(self.root)
        frm.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
        ttk.Label(frm, textvariable=self.beamline, style="title.TLabel").pack(side=tk.RIGHT)

        frm = ttk.Frame(self.root)
        frm.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
        ttk.Label(frm, text='Visit:').pack(side=tk.LEFT, padx=4)
        self.visit_menu = ttk.OptionMenu(frm, self.visit, *list(self.visits.keys()), command=self.choose_visit)
        self.visit_menu.pack(side=tk.LEFT, padx=4)
        ttk.Button(frm, text='Check', command=self.open_file_browser, width=10).pack(side=tk.LEFT)

        frm = ttk.Frame(self.root)
        frm.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=4)
        ttk.Label(frm, text='Data Dir:', width=15).pack(side=tk.LEFT, padx=4)
        ttk.Entry(frm, textvariable=self.data_dir, width=60).pack(side=tk.LEFT)
        ttk.Button(frm, text='Browse', command=self.browse_datadir).pack(side=tk.LEFT)

        frm = ttk.Frame(self.root)
        frm.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=4)
        ttk.Label(frm, text='Analysis Dir:', width=15).pack(side=tk.LEFT, padx=4)
        ttk.Entry(frm, textvariable=self.proc_dir, width=60).pack(side=tk.LEFT)
        ttk.Button(frm, text='Browse', command=self.browse_analysis).pack(side=tk.LEFT)

        frm = ttk.Frame(self.root)
        frm.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=4)
        ttk.Label(frm, text='Notebook Dir:', width=15).pack(side=tk.LEFT, padx=4)
        ttk.Entry(frm, textvariable=self.notebook_dir, width=60).pack(side=tk.LEFT)
        ttk.Button(frm, text='Browse', command=self.browse_notebook).pack(side=tk.LEFT)

        frm = ttk.Frame(self.root)
        frm.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, pady=6)
        ttk.Button(frm, text='Data Viewer', command=self.open_data_viewer, width=20).pack(side=tk.LEFT)
        # ttk.Button(frm, text='NeXus Browser', command=self.open_file_browser, width=20).pack(side=tk.LEFT)
        ttk.Button(frm, text='Log Viewer', command=self.open_log_viewer, width=20).pack(side=tk.LEFT)
        ttk.Button(frm, text='Notebook Browser', command=self.open_notebook_browser, width=20).pack(side=tk.LEFT)
        ttk.Button(frm, text='Script Runner', command=self.open_script_runner, width=20).pack(side=tk.LEFT)

        self.choose_beamline(self.config.get(C.beamline, 'i16'))

    def choose_beamline(self, beamline: str):
        bl_config = get_config(beamline=beamline)
        self.config.update(bl_config)
        self.beamline.set('MMG Toolbox: ' + beamline)
        self.visits = get_dls_visits(beamline)
        self.visits.update({'default': self.config.get(C.default_directory, '.')})
        current_visit = next(iter(self.visits.keys()))
        self.visit_menu.set_menu(*self.visits)
        self.visit.set(current_visit)
        self.dls_directories(self.visits[current_visit])

    def menu_items(self):

        menu = {
            'Recent Files': {
                file: lambda x=file: self.data_dir.set(x)
                for file in self.config.get(C.recent_data_directories)
            },
            'Beamline': {
                bl: lambda x=bl: self.choose_beamline(x)
                for bl in MMG_BEAMLINES
            },
        }
        return menu

    def dls_directories(self, data_dir: str):
        if not os.access(data_dir, os.R_OK):
            show_error(f"Warning path is not readable: '{data_dir}'", self.root, raise_exception=False)
        self.data_dir.set(data_dir)
        proc_dir = os.path.join(data_dir, 'processing')
        notebook_dir = os.path.join(data_dir, 'processed', 'notebooks')
        if os.path.isdir(proc_dir) and os.access(proc_dir, os.W_OK):
            self.proc_dir.set(proc_dir)
        if os.path.isdir(notebook_dir) and os.access(notebook_dir, os.R_OK):
            self.notebook_dir.set(notebook_dir)

    def update_config(self):
        save_config(self.config)

    def add_recent_directory(self, directory: str):
        recent = self.config.get(C.recent_data_directories, [])
        if directory in recent:
            return
        recent.insert(0, directory)
        while len(recent) > 10:
            recent.pop()
        self.update_config()

    def choose_visit(self, event=None):
        visit_folder = self.visits[self.visit.get()]
        self.dls_directories(visit_folder)

    def browse_datadir(self):
        folder = select_folder(self.root)
        if folder:
            self.dls_directories(folder)

    def browse_analysis(self):
        folder = select_folder(self.root)
        if folder:
            self.proc_dir.set(folder)

    def browse_notebook(self):
        folder = select_folder(self.root)
        if folder:
            self.notebook_dir.set(folder)

    def open_data_viewer(self):
        from ..main import create_data_viewer
        self.add_recent_directory(self.data_dir.get())
        create_data_viewer(self.data_dir.get(), self.root, self.config)

    def open_file_browser(self):
        from ..main import create_nexus_file_browser
        create_nexus_file_browser(self.root, self.data_dir.get())

    def open_log_viewer(self):
        from .log_viewer import create_gda_terminal_log_viewer
        create_gda_terminal_log_viewer(self.data_dir.get(), self.root)

    def open_notebook_browser(self):
        from ..main import create_jupyter_browser
        create_jupyter_browser(self.root, self.notebook_dir.get())

    def open_script_runner(self):
        from ..main import create_script_runner
        folders = {
            C.default_directory: self.data_dir.get(),
            C.processing_directory: self.proc_dir.get(),
            C.notebook_directory: self.notebook_dir.get(),
        }
        self.config.update(folders)
        create_script_runner(self.root, self.config)




