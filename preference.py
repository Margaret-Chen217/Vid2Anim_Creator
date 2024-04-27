from typing import List

import bpy
from . import setup_utils

_LINES: List[str] = []


def _lines_append(line: str):
    if line.startswith("\r") and len(_LINES) > 0:
        del _LINES[-1]
        line = line[1:]
    _LINES.append(line)


class UpdatePythonModulesOperator(bpy.types.Operator):
    bl_idname = "motion_generate_tools.update_python_modules"
    bl_label = "Update Python Modules"
    bl_options = {"REGISTER", "INTERNAL"}

    use_gpu: bpy.props.BoolProperty(default=False)

    @classmethod
    def poll(cls, _context):
        return not setup_utils.is_installer_running()

    def execute(self, context):
        _LINES.clear()
        region = context.region
        setup_utils.install_python_modules(
            use_gpu=self.use_gpu,
            line_callback=lambda line: _lines_append(line) or region.tag_redraw(),
            finally_callback=lambda e: region.tag_redraw(),
        )
        return {"FINISHED"}


class UninstallPythonModulesOperator(bpy.types.Operator):
    bl_idname = "motion_generate_tools.uninstall_python_modules"
    bl_label = "Uninstall Python Modules"
    bl_options = {"REGISTER", "INTERNAL"}

    @classmethod
    def poll(cls, _context):
        return not setup_utils.is_installer_running()

    def execute(self, context):
        _LINES.clear()
        region = context.region
        setup_utils.uninstall_python_modules(
            line_callback=lambda line: _lines_append(line) or region.tag_redraw(),
            finally_callback=lambda e: region.tag_redraw(),
        )
        return {"FINISHED"}


class ListPythonModulesOperator(bpy.types.Operator):
    bl_idname = "motion_generate_tools.list_python_modules"
    bl_label = "List Python Modules"
    bl_options = {"REGISTER", "INTERNAL"}

    @classmethod
    def poll(cls, _context):
        return not setup_utils.is_installer_running()

    def execute(self, context):
        _LINES.clear()
        region = context.region
        setup_utils.list_python_modules(
            line_callback=lambda line: _lines_append(line) or region.tag_redraw(),
            finally_callback=lambda e: region.tag_redraw(),
        )
        return {"FINISHED"}


class Vid2AnimCreatorAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    show_logs: bpy.props.BoolProperty(default=False)

    def draw(self, _context: bpy.types.Context):
        layout = self.layout

        col = layout.column(align=True)
        flow = col.grid_flow(align=True)
        flow.row().label(text="Required Python Modules:")
        for name, is_installed in setup_utils.get_required_modules().items():
            flow.row().label(text=name, icon="CHECKMARK" if is_installed else "ERROR")
        flow = col.grid_flow(align=True)
        row = flow.row(align=True)
        row.operator(UpdatePythonModulesOperator.bl_idname).use_gpu = False
        row.operator(
            UpdatePythonModulesOperator.bl_idname, text="with CUDA"
        ).use_gpu = True
        flow.row().operator(UninstallPythonModulesOperator.bl_idname)
        flow.row().operator(ListPythonModulesOperator.bl_idname)

        col = layout.column(align=True)
        flow = col.grid_flow(align=True)
        col = layout.column(align=False)
        row = col.row(align=True)
        row.prop(
            self,
            "show_logs",
            icon="TRIA_DOWN" if self.show_logs else "TRIA_RIGHT",
            icon_only=True,
            emboss=False,
        )
        row.label(text="Logs")
        exit_code = setup_utils.get_installer_exit_code()
        if setup_utils.is_installer_running():
            row.label(text="Processing ...", icon="SORTTIME")
        elif exit_code >= 0:
            row.label(
                text=f"Done with code: {exit_code}",
                icon="CHECKMARK" if exit_code == 0 else "ERROR",
            )

        if self.show_logs:
            box = col.box().column(align=True)
            for line in _LINES:
                box.label(text=line)
