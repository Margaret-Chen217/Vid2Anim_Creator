import os
import sys
import shutil
import pickle as pk

sys.path.append(
    "D:\\FinalProject\\BlenderDevelopStudy\\blenderAddOn\\folder\\hybrik_blender_addon"
)


# import blender python package
import bpy
from bpy.types import Context
from bpy_extras.io_utils import ImportHelper


from . import predict, util, preference

# set temp folder path
project_path = os.path.expanduser("~")
tempfolder_path = os.path.join(project_path, "temp")

if os.path.exists(tempfolder_path):
    shutil.rmtree(tempfolder_path)

os.makedirs(tempfolder_path)


class PG_DenoisePropsGroup(bpy.types.PropertyGroup):
    denoise_mode: bpy.props.EnumProperty(
        name="mode",
        description="denoise mode",
        items=[
            ("butterworth", "butterworth", ""),
            ("conv_avg", "conv_avg", ""),
            ("conv_gaussian", "conv_gaussian", ""),
        ],
    )


class PG_ButterWorthPropsGroup(bpy.types.PropertyGroup):
    order: bpy.props.IntProperty(
        name="order", description="butter worth order", default=4
    )

    cutoff_freq: bpy.props.FloatProperty(
        name="cutoff_freq", description="cutoff frequency", default=0.1
    )

    sampling_freq: bpy.props.IntProperty(
        name="sampling_freq", description="sample frequency", default=1
    )


class PG_AvgConvPropsGroup(bpy.types.PropertyGroup):
    kernel_size: bpy.props.IntProperty(
        name="kernel size", description="kernel size", default=3
    )


class PG_AvgGaussianPropsGroup(bpy.types.PropertyGroup):
    sigma: bpy.props.FloatProperty(name="sigma", description="sigma", default=1.0)


bl_info = {
    "name": "Vid2Anim Creator",
    "author": "MargaretChen",
    "description": "Monocular Video to Character Animation Tool",
    "blender": (2, 80, 0),
    "version": (0, 0, 1),
    "location": "",
    "warning": "",
    "category": "Animation",
    "tracker_url": "https://github.com/Margaret-Chen217/hybrik_blender_addon",
}


def initProperty():
    bpy.types.Scene.sourcefile_path = bpy.props.StringProperty(
        name="", description="Some tooltip", default=""
    )
    bpy.types.Scene.use_gpu = bpy.props.BoolProperty(
        name="use_gpu", description="Some tooltip", default=True
    )
    bpy.types.Scene.use_denoise = bpy.props.BoolProperty(
        name="use_denoise", description="Some tooltip", default=False
    )


class GenerateAnimPanel(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""

    bl_label = "Convert2Anim"
    bl_idname = "ANIM_PT_layout"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "V2A"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Source File
        layout.label(text="Source File")
        row = layout.row(align=True)
        row.prop(scene, "sourcefile_path")
        row.operator("hybrik.import_source_file", text="", icon="IMPORT")

        # GPU Setting
        layout.label(text="GPU Setting")
        row = layout.row(align=True)
        row.prop(scene, "use_gpu")

        # Denoise
        layout.label(text="Denoise Setting")
        row = layout.row(align=True)
        row.prop(scene, "use_denoise")
        if context.scene.use_denoise:
            row = layout.row(align=True)
            row.prop(context.window_manager.anim_tool, "denoise_mode")

            if context.window_manager.anim_tool.denoise_mode == "butterworth":
                row = layout.row(align=True)
                row.label(text="order")
                row.prop(context.window_manager.butterworth, "order", text="")

                row = layout.row(align=True)
                row.label(text="cutoff_freq")
                row.prop(context.window_manager.butterworth, "cutoff_freq", text="")

                row = layout.row(align=True)
                row.label(text="sampling_freq")
                row.prop(context.window_manager.butterworth, "sampling_freq", text="")

            if context.window_manager.anim_tool.denoise_mode == "conv_avg":
                row = layout.row(align=True)
                row.label(text="kernel size")
                row.prop(context.window_manager.conv_avg, "kernel_size", text="")

            if context.window_manager.anim_tool.denoise_mode == "conv_gaussian":
                row = layout.row(align=True)
                row.label(text="sigma")
                row.prop(context.window_manager.conv_gaussian, "sigma", text="")

        # Hybrik
        layout.label(text="Animation")
        row = layout.row(align=True)
        row.operator("hybrik.generate_hybrik_anim", text="Hybrik")
        row.operator("hybrik.generate_niki_anim", text="Niki")


class UtilPanel(bpy.types.Panel):
    bl_label = "Util"
    bl_idname = "UTIL_PT_layout"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "V2A"

    def draw(self, context: Context):
        layout = self.layout
        scene = context.scene

        # Util
        layout.label(text="Load pk")
        row = layout.row(align=True)
        row.operator("hybrik.load_pk", text="import .pk file")

        # Hybrik
        layout.label(text="Smooth")
        row = layout.row(align=True)
        row.operator("hybrik.import_source_file", text="SMOOTH ANIMATION")
        # layout.label(text=f"Progress: {context.scene.progress}")

        # Reset Location
        layout.label(text="Reset Location")
        row = layout.row(align=True)
        row.operator("hybrik.reset_location", text="RESET LOCATION")

        # Export
        layout.label(text="Export")
        row = layout.row(align=True)
        row.operator("export_scene.fbx", text="FBX")
        row.operator("export_scene.fbx", text="BVH")
        row.operator("export_scene.fbx", text="PK")


class ImportSourceFileOperator(bpy.types.Operator, ImportHelper):
    bl_idname = "hybrik.import_source_file"
    bl_label = "Import"
    bl_description = "Opens file selector, after executes"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        fdir = self.properties.filepath
        # save filepath
        context.scene.sourcefile_path = fdir

        if os.path.splitext(fdir)[1].lower() == ".mp4":
            # is mp4 file
            self.report({"INFO"}, "Import file:" + fdir)
            return {"FINISHED"}
        else:
            # not mp4 file
            self.report({"ERROR"}, "File Format ERROR")
            return {"CANCELLED"}


class GenerateHybrikAnimOperator(bpy.types.Operator):
    bl_idname = "hybrik.generate_hybrik_anim"
    bl_label = ""
    bl_description = "Description that shows in blender tooltips"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        img_path_list = util.video2imageSeq(context, tempfolder_path)

        denoise = context.scene.use_denoise
        if denoise:
            print("denoise mode: ", context.window_manager.anim_tool.denoise_mode)
            denoise = context.window_manager.anim_tool.denoise_mode
        else:
            denoise = None

        hybrik_ins = predict.Hybrik(context)
        hybrik_ins.prepare()
        res_db = hybrik_ins.run(img_path_list, denoise)

        # save pk file
        # with open(os.path.join(tempfolder_path, "res.pk"), "wb") as fid:
        #     pk.dump(res_db, fid)

        root_path = os.path.dirname(os.path.realpath(__file__))
        util.load_bvh(self, res_db, root_path)

        return {"FINISHED"}


class GenerateNikiAnimOperator(bpy.types.Operator):
    bl_idname = "hybrik.generate_niki_anim"
    bl_label = ""
    bl_description = "Description that shows in blender tooltips"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        img_path_list = util.video2imageSeq(context, tempfolder_path)
        denoise = context.scene.use_denoise
        niki_ins = predict.Niki(context)
        niki_ins.prepare()
        res_db = niki_ins.run(img_path_list, denoise)
        print("Niki Test Finished")

        root_path = os.path.dirname(os.path.realpath(__file__))
        util.load_bvh(self, res_db, root_path)

        return {"FINISHED"}


class ResetLocationOperator(bpy.types.Operator):
    bl_idname = "hybrik.reset_location"
    bl_label = "Reset locations"
    bl_description = "Reset locations"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        frame_end = bpy.context.scene.frame_end
        gender = "m"
        if "SMPLX-neutral" in bpy.data.objects:
            arm_obname = "SMPLX-neutral"
            prefix = ""
        else:
            obname = "%s_avg" % gender[0]
            arm_obname = "Armature"
            prefix = obname + "_"

        arm_ob = bpy.data.objects[arm_obname]
        for fid in range(frame_end):
            arm_ob.pose.bones[prefix + "root"].location = [0, 0, 0]
            arm_ob.pose.bones[prefix + "root"].keyframe_insert("location", frame=fid)

        return {"FINISHED"}


class LoadPKOperator(bpy.types.Operator, ImportHelper):
    bl_idname = "hybrik.load_pk"
    bl_label = "Load .pk file"
    bl_description = "Load pk file"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: Context):
        fdir = self.properties.filepath
        with open(fdir, "rb") as fid:
            res_db = pk.load(fid)

        root_path = os.path.dirname(os.path.realpath(__file__))
        util.load_bvh(self, res_db, root_path)

        return {"FINISHED"}


classes = [
    preference.ListPythonModulesOperator,
    preference.UninstallPythonModulesOperator,
    preference.UpdatePythonModulesOperator,
    preference.Vid2AnimCreatorAddonPreferences,
    LoadPKOperator,
    GenerateAnimPanel,
    UtilPanel,
    PG_DenoisePropsGroup,
    ImportSourceFileOperator,
    GenerateHybrikAnimOperator,
    ResetLocationOperator,
    GenerateNikiAnimOperator,
    PG_ButterWorthPropsGroup,
    PG_AvgConvPropsGroup,
    PG_AvgGaussianPropsGroup,
]


def register():
    initProperty()
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.WindowManager.anim_tool = bpy.props.PointerProperty(
        type=PG_DenoisePropsGroup
    )
    bpy.types.WindowManager.butterworth = bpy.props.PointerProperty(
        type=PG_ButterWorthPropsGroup
    )
    bpy.types.WindowManager.conv_avg = bpy.props.PointerProperty(
        type=PG_AvgConvPropsGroup
    )
    bpy.types.WindowManager.conv_gaussian = bpy.props.PointerProperty(
        type=PG_AvgGaussianPropsGroup
    )


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    del bpy.types.WindowManager.anim_tool
    del bpy.types.WindowManager.butterworth
    del bpy.types.WindowManager.conv_avg
    del bpy.types.WindowManager.conv_gaussian


if __name__ == "__main__":
    register()
