import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import seed as seed_helper
from replicate_weights import download_replicate_weights
from dataclasses import dataclass

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")
api_json_file = "workflow.json"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

@dataclass
class Inputs:
    prompt = Input(description="Text prompt for video generation")
    negative_prompt = Input(description="Things you do not want to see in your video", default="")
    aspect_ratio = Input(description="The aspect ratio of the video. 16:9, 9:16, 1:1, etc.", choices=["16:9", "9:16", "1:1"], default="16:9")
    frames = Input(description="The number of frames to generate (1 to 5 seconds)", choices=[17, 33, 49, 65, 81], default=81)
    model = Input(description="The model to use. 1.3b is faster, but 14b is better quality. A LORA either works with 1.3b or 14b, depending on the version it was trained on.", choices=["1.3b", "14b"], default="14b")
    lora_url = Input(description="Optional: The URL of a LORA to use", default=None)
    lora_strength_model = Input(description="Strength of the LORA applied to the model. 0.0 is no LORA.", default=1.0)
    lora_strength_clip = Input(description="Strength of the LORA applied to the CLIP model. 0.0 is no LORA.", default=1.0)
    sample_shift = Input(description="Sample shift factor", default=8.0, ge=0.0, le=10.0)
    sample_guide_scale = Input(description="Higher guide scale makes prompt adherence better, but can reduce variation", default=5.0, ge=0.0, le=10.0)
    sample_steps = Input(description="Number of generation steps. Fewer steps means faster generation, at the expensive of output quality. 30 steps is sufficient for most prompts", default=30, ge=1, le=60)
    seed = Input(default=seed_helper.predict_seed())
    replicate_weights = Input(description="Replicate LoRA weights to use. Leave blank to use the default weights.", default=None)

class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[
                "wan2.1_t2v_1.3B_bf16.safetensors",
                "wan2.1_t2v_14B_bf16.safetensors",
                "wan_2.1_vae.safetensors",
                "umt5_xxl_fp16.safetensors",
            ],
        )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    def update_workflow(self, workflow, **kwargs):
        empty_latent_video = workflow["40"]["inputs"]
        empty_latent_video["length"] = kwargs["frames"]
        if kwargs["aspect_ratio"] == "16:9":
            empty_latent_video["width"] = 832
            empty_latent_video["height"] = 480
        elif kwargs["aspect_ratio"] == "9:16":
            empty_latent_video["width"] = 480
            empty_latent_video["height"] = 832

        model_loader = workflow["37"]["inputs"]
        if kwargs["model"] == "1.3b":
            model_loader["unet_name"] = "wan2.1_t2v_1.3B_bf16.safetensors"
        elif kwargs["model"] == "14b":
            model_loader["unet_name"] = "wan2.1_t2v_14B_bf16.safetensors"

        positive_prompt = workflow["6"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        negative_prompt = workflow["7"]["inputs"]
        negative_prompt["text"] = f"nsfw, {kwargs['negative_prompt']}"

        sampler = workflow["3"]["inputs"]
        sampler["seed"] = kwargs["seed"]
        sampler["cfg"] = kwargs["sample_guide_scale"]
        sampler["steps"] = kwargs["sample_steps"]

        shift = workflow["48"]["inputs"]
        shift["shift"] = kwargs["sample_shift"]

        if kwargs["lora_url"] or kwargs["lora_filename"]:
            lora_loader = workflow["49"]["inputs"]
            if kwargs["lora_filename"]:
                lora_loader["lora_name"] = kwargs["lora_filename"]
            elif kwargs["lora_url"]:
                lora_loader["lora_name"] = kwargs["lora_url"]

            lora_loader["strength_model"] = kwargs["lora_strength_model"]
            lora_loader["strength_clip"] = kwargs["lora_strength_clip"]
        else:
            del workflow["49"]
            positive_prompt["clip"] = ["38", 0]
            shift["model"] = ["54", 0]

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        aspect_ratio: str = "16:9",
        frames: int = 81,
        model: str = "14b",
        lora_url: str | None = None,
        lora_strength_model: float = 1.0,
        lora_strength_clip: float = 1.0,
        sample_shift: float = 8.0,
        sample_guide_scale: float = 5.0,
        sample_steps: int = 30,
        seed: int | None = None,
        replicate_weights: str | None = None,
    ) -> List[Path]:
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        seed = seed_helper.generate(seed)

        lora_filename = None
        if replicate_weights:
            lora_filename = download_replicate_weights(replicate_weights, "ComfyUI/models/loras")

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            sample_shift=sample_shift,
            sample_guide_scale=sample_guide_scale,
            sample_steps=sample_steps,
            model=model,
            frames=frames,
            aspect_ratio=aspect_ratio,
            lora_filename=lora_filename,
            lora_url=lora_url,
            lora_strength_model=lora_strength_model,
            lora_strength_clip=lora_strength_clip,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return self.comfyUI.get_files(OUTPUT_DIR, file_extensions=["mp4"])

class StandaloneLoraPredictor(Predictor):
    def predict(self,
            prompt: str = Inputs.prompt,
            negative_prompt: str = Inputs.negative_prompt,
            aspect_ratio: str = Inputs.aspect_ratio,
            frames: int = Inputs.frames,
            model: str = Inputs.model,
            lora_url: str = Inputs.lora_url,
            lora_strength_model: float = Inputs.lora_strength_model,
            lora_strength_clip: float = Inputs.lora_strength_clip,
            sample_shift: float = Inputs.sample_shift,
            sample_guide_scale: float = Inputs.sample_guide_scale,
            sample_steps: int = Inputs.sample_steps,
            seed: int = seed_helper.predict_seed(),
        ) -> List[Path]:
        return self.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            frames=frames,
            model=model,
            lora_url=lora_url,
            lora_strength_model=lora_strength_model,
            lora_strength_clip=lora_strength_clip,
            sample_shift=sample_shift,
            sample_guide_scale=sample_guide_scale,
            sample_steps=sample_steps,
            seed=seed,
            replicate_weights=None,
        )

class Trained14BLoraPredictor(Predictor):
    def predict(self,
            prompt: str = Inputs.prompt,
            negative_prompt: str = Inputs.negative_prompt,
            aspect_ratio: str = Inputs.aspect_ratio,
            frames: int = Inputs.frames,
            lora_strength_model: float = Inputs.lora_strength_model,
            lora_strength_clip: float = Inputs.lora_strength_clip,
            sample_shift: float = Inputs.sample_shift,
            sample_guide_scale: float = Inputs.sample_guide_scale,
            sample_steps: int = Inputs.sample_steps,
            seed: int = seed_helper.predict_seed(),
            replicate_weights: str = Inputs.replicate_weights,
        ) -> List[Path]:
        return self.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            frames=frames,
            model="14b",
            lora_url=None,
            lora_strength_model=lora_strength_model,
            lora_strength_clip=lora_strength_clip,
            sample_shift=sample_shift,
            sample_guide_scale=sample_guide_scale,
            sample_steps=sample_steps,
            seed=seed,
            replicate_weights=replicate_weights,
        )
