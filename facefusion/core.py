import os

os.environ['OMP_NUM_THREADS'] = '1'

import sys
import platform
import shutil
import onnxruntime

import facefusion.choices
import facefusion.globals
from facefusion.face_analyser import get_one_face
from facefusion.face_reference import get_face_reference, set_face_reference
from facefusion.vision import read_image
from facefusion import face_analyser, wording
from facefusion.processors.frame.core import get_frame_processors_modules
from facefusion.processors.frame import globals as frame_processors_globals
from facefusion.utilities import is_image, decode_execution_providers, normalize_output_path, update_status

onnxruntime.set_default_logger_severity(3)


def run(source_path, target_path, output_path):
    apply_args(source_path, target_path, output_path)
    limit_resources()
    if not pre_check() or not face_analyser.pre_check():
        return None
    for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
        if not frame_processor_module.pre_check():
            return None
    conditional_process()
    if is_image(output_path):
        return output_path
    return None


def apply_args(source_path, target_path, output_path, image_quality=100) -> None:
    # general
    facefusion.globals.source_path = source_path
    facefusion.globals.target_path = target_path
    facefusion.globals.output_path = normalize_output_path(facefusion.globals.source_path,
                                                           facefusion.globals.target_path, output_path)
    # misc
    facefusion.globals.skip_download = False
    # execution
    facefusion.globals.execution_providers = decode_execution_providers(['cpu'])
    print(f"[ FACE FUSION ] device use {facefusion.globals.execution_providers}")
    facefusion.globals.execution_thread_count = 1
    facefusion.globals.execution_queue_count = 1
    facefusion.globals.max_memory = None
    # face analyser
    facefusion.globals.face_analyser_order = 'large-small'
    facefusion.globals.face_analyser_age = None
    facefusion.globals.face_analyser_gender = None
    facefusion.globals.face_detector_model = 'retinaface'
    facefusion.globals.face_detector_size = '640x640'
    facefusion.globals.face_detector_score = 0.72
    # face selector
    facefusion.globals.face_selector_mode = 'one'
    facefusion.globals.reference_face_position = 0
    facefusion.globals.reference_face_distance = 0.6
    facefusion.globals.reference_frame_number = 0
    # face mask
    facefusion.globals.face_mask_blur = 0.3
    facefusion.globals.face_mask_padding = (0, 0, 0, 0)
    # output creation
    facefusion.globals.output_image_quality = image_quality
    # frame processors
    facefusion.globals.frame_processors = ['face_swapper', 'face_enhancer']
    frame_processors_globals.face_swapper_model = "inswapper_128"
    facefusion.globals.face_recognizer_model = 'arcface_inswapper'
    frame_processors_globals.face_enhancer_model = 'gfpgan_1.4'
    frame_processors_globals.face_enhancer_blend = 100


def limit_resources() -> None:
    if facefusion.globals.max_memory:
        memory = facefusion.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = facefusion.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32 # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status(wording.get('python_not_supported').format(version = '3.9'))
        return False
    return True


def conditional_process() -> None:
    conditional_set_face_reference()
    for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
        if not frame_processor_module.pre_process('output'):
            return
    if is_image(facefusion.globals.target_path):
        process_image()


def conditional_set_face_reference() -> None:
    if 'reference' in facefusion.globals.face_selector_mode and not get_face_reference():
        if is_image(facefusion.globals.target_path):
            reference_frame = read_image(facefusion.globals.target_path)
            reference_face = get_one_face(reference_frame, facefusion.globals.reference_face_position)
            set_face_reference(reference_face)


def process_image() -> None:
    shutil.copy2(facefusion.globals.target_path, facefusion.globals.output_path)
    # process frame
    for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
        update_status(wording.get('processing'), frame_processor_module.NAME)
        frame_processor_module.process_image(facefusion.globals.source_path, facefusion.globals.output_path, facefusion.globals.output_path)
        frame_processor_module.post_process()
    # validate image
    if is_image(facefusion.globals.output_path):
        update_status(wording.get('processing_image_succeed'))
    else:
        update_status(wording.get('processing_image_failed'))
