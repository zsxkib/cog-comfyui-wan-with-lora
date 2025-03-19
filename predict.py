import os
import mimetypes
import json
import shutil
import re
import time
import statistics
import platform
from typing import List, Dict, Any, Optional, Tuple
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import seed as seed_helper
from replicate_weights import download_replicate_weights
from dataclasses import dataclass
import torch

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
COMFYUI_LORAS_DIR = "ComfyUI/models/loras"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")
api_json_file = "workflow.json"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


@dataclass
class Inputs:
    prompt = Input(description="Text prompt for video generation", default="astronaut riding a unicorn on mars")
    negative_prompt = Input(
        description="Things you do not want to see in your video", default=""
    )
    image = Input(
        description="Image to use as a starting frame for image to video generation.",
        default=None,
    )
    aspect_ratio = Input(
        description="The aspect ratio of the video. 16:9, 9:16, 1:1, etc.",
        choices=["16:9", "9:16", "1:1"],
        default="16:9",
    )
    frames = Input(
        description="The number of frames to generate (1 to 5 seconds)",
        choices=[17, 33, 49, 65, 81],
        default=81,
    )
    model = Input(
        description="The model to use. 1.3b is faster, but 14b is better quality. A LORA either works with 1.3b or 14b, depending on the version it was trained on.",
        choices=["1.3b", "14b"],
        default="14b",
    )
    lora_url = Input(description="Optional: The URL of a LORA to use", default=None)
    lora_strength_model = Input(
        description="Strength of the LORA applied to the model. 0.0 is no LORA.",
        default=1.0,
    )
    lora_strength_clip = Input(
        description="Strength of the LORA applied to the CLIP model. 0.0 is no LORA.",
        default=1.0,
    )
    sample_shift = Input(
        description="Sample shift factor", default=8.0, ge=0.0, le=10.0
    )
    sample_guide_scale = Input(
        description="Higher guide scale makes prompt adherence better, but can reduce variation",
        default=5.0,
        ge=0.0,
        le=10.0,
    )
    sample_steps = Input(
        description="Number of generation steps. Fewer steps means faster generation, at the expensive of output quality. 30 steps is sufficient for most prompts",
        default=30,
        ge=1,
        le=60,
    )
    seed = Input(default=seed_helper.predict_seed())
    replicate_weights = Input(
        description="Replicate LoRA weights to use. Leave blank to use the default weights.",
        default=None,
    )
    fast_mode = Input(
        description="Speed up generation with different levels of acceleration. Faster modes may degrade quality somewhat. The speedup is dependent on the content, so different videos may see different speedups.",
        choices=["Off", "Balanced", "Fast"],
        default="Balanced",
    )
    resolution = Input(
        description="The resolution of the video. 720p is not supported for 1.3b.",
        choices=["480p", "720p"],
        default="480p",
    )
    benchmark = Input(
        description="Run generation multiple times and report timing statistics",
        default=False,
    )
    benchmark_runs = Input(
        description="Number of benchmark runs if benchmarking is enabled",
        default=5,
        ge=1,
        le=100,
    )


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        os.makedirs("ComfyUI/models/loras", exist_ok=True)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[
                "wan_2.1_vae.safetensors",
                "umt5_xxl_fp16.safetensors",
                "clip_vision_h.safetensors",
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

    def get_width_and_height(self, resolution: str, aspect_ratio: str):
        sizes = {
            "480p": {
                "16:9": (832, 480),
                "9:16": (480, 832),
                "1:1": (644, 644),
            },
            "720p": {
                "16:9": (1280, 720),
                "9:16": (720, 1280),
                "1:1": (980, 980),
            },
        }
        return sizes[resolution][aspect_ratio]

    def set_model_loader(self, model_loader, model: str):
        if model == "14b-i2v-480p":
            model_loader["unet_name"] = "wan2.1_i2v_480p_14B_bf16.safetensors"
        elif model == "14b-i2v-720p":
            model_loader["unet_name"] = "wan2.1_i2v_720p_14B_bf16.safetensors"
        elif model == "14b":
            model_loader["unet_name"] = "wan2.1_t2v_14B_bf16.safetensors"
        elif model == "1.3b":
            model_loader["unet_name"] = "wan2.1_t2v_1.3B_bf16.safetensors"

    def update_workflow(self, workflow, **kwargs):
        is_image_to_video = kwargs["image_filename"] is not None
        model = f"{kwargs['model']}-i2v-{kwargs['resolution']}" if is_image_to_video else kwargs["model"]

        self.set_model_loader(workflow["37"]["inputs"], model)

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

        if is_image_to_video:
            del workflow["40"]
            wan_i2v_latent = workflow["58"]["inputs"]
            wan_i2v_latent["length"] = kwargs["frames"]

            image_loader = workflow["55"]["inputs"]
            image_loader["image"] = kwargs["image_filename"]

            image_resizer = workflow["56"]["inputs"]
            if kwargs["resolution"] == "720p":
                image_resizer["target_size"] = 1008
            else:
                image_resizer["target_size"] = 644

        else:
            del workflow["55"]
            del workflow["56"]
            del workflow["57"]
            del workflow["58"]
            del workflow["59"]
            del workflow["60"]
            width, height = self.get_width_and_height(
                kwargs["resolution"], kwargs["aspect_ratio"]
            )
            empty_latent_video = workflow["40"]["inputs"]
            empty_latent_video["length"] = kwargs["frames"]
            empty_latent_video["width"] = width
            empty_latent_video["height"] = height

            sampler["model"] = ["48", 0]
            sampler["positive"] = ["6", 0]
            sampler["negative"] = ["7", 0]
            sampler["latent_image"] = ["40", 0]

        thresholds = {
            "14b": {
                "Balanced": 0.15,
                "Fast": 0.2,
                "coefficients": "14B",
            },
            "14b-i2v-480p": {
                "Balanced": 0.19,
                "Fast": 0.26,
                "coefficients": "i2v_480",
            },
            "14b-i2v-720p": {
                "Balanced": 0.2,
                "Fast": 0.3,
                "coefficients": "i2v_720",
            },
            "1.3b": {
                "Balanced": 0.07,
                "Fast": 0.08,
                "coefficients": "1.3B",
            },
        }

        fast_mode = kwargs["fast_mode"]
        if fast_mode == "Off":
            # Turn off tea cache
            del workflow["54"]
            workflow["49"]["inputs"]["model"] = ["37", 0]
        else:
            tea_cache = workflow["54"]["inputs"]
            tea_cache["coefficients"] = thresholds[model]["coefficients"]
            tea_cache["rel_l1_thresh"] = thresholds[model][fast_mode]

        if kwargs["lora_url"] or kwargs["lora_filename"]:
            lora_loader = workflow["49"]["inputs"]
            if kwargs["lora_filename"]:
                lora_loader["lora_name"] = kwargs["lora_filename"]
            elif kwargs["lora_url"]:
                lora_loader["lora_name"] = kwargs["lora_url"]

            lora_loader["strength_model"] = kwargs["lora_strength_model"]
            lora_loader["strength_clip"] = kwargs["lora_strength_clip"]
        else:
            del workflow["49"]  # delete lora loader node
            positive_prompt["clip"] = ["38", 0]
            shift["model"] = ["37", 0] if fast_mode == "Off" else ["54", 0]

    def benchmark_generate(
        self, 
        benchmark_runs: int, 
        model_info: str, 
        resolution: str, 
        frames: int, 
        fast_mode: str,
        generate_kwargs: dict
    ) -> List[Path]:
        """Run benchmarking for the generate method"""
        # Get GPU information
        gpu_info = "Unknown GPU"
        gpu_memory_gb = 0
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        
        # Determine if we're in image-to-video mode
        is_i2v = generate_kwargs.get("image") is not None
        mode_emoji = "🖼️➡️🎬" if is_i2v else "📝➡️🎬"
        
        # Get image dimensions if in i2v mode
        image_dims = None
        if is_i2v and generate_kwargs.get("image"):
            try:
                from PIL import Image
                img = Image.open(generate_kwargs["image"])
                image_dims = f"{img.width}x{img.height}"
            except Exception:
                image_dims = "Unknown"
        
        print(f"\n{'='*80}")
        print(f"{mode_emoji} RUNNING BENCHMARK: {benchmark_runs} iterations")
        print(f"Configuration: {model_info}, Resolution: {resolution}, Frames: {frames}, Fast Mode: {fast_mode}")
        if is_i2v and image_dims:
            print(f"Input Image: {image_dims}")
        print(f"GPU: {gpu_info} ({gpu_memory_gb:.2f} GB)")
        print(f"{'='*80}\n")
        
        execution_times = []
        memory_usage = []
        result = None
        seed = generate_kwargs.get('seed')
        start_total_time = time.time()
        
        for i in range(benchmark_runs):
            print(f"Run {i+1}/{benchmark_runs}...")
            
            # Record initial GPU memory usage
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            
            start_time = time.time()
            
            # Use the original seed for the first run, but different seeds for subsequent runs
            if i > 0:
                # Make sure we're generating an actual integer for the seed, not using the FieldInfo
                current_kwargs = generate_kwargs.copy()
                current_kwargs['seed'] = seed_helper.predict_seed()
            else:
                current_kwargs = generate_kwargs
            
            result = self.generate(**current_kwargs)
            
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            # Record peak GPU memory usage
            memory_used = 0
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
                memory_used = peak_memory - initial_memory
                memory_usage.append(memory_used)
                print(f"Run {i+1} completed in {execution_time:.2f} seconds (Peak GPU memory: {peak_memory:.2f} GB)")
            else:
                print(f"Run {i+1} completed in {execution_time:.2f} seconds")
                
            # Calculate running statistics
            current_avg_time = statistics.mean(execution_times)
            current_median_time = statistics.median(execution_times)
            if len(execution_times) > 1:
                current_stdev_time = statistics.stdev(execution_times)
            else:
                current_stdev_time = 0
                
            # Print running averages
            print(f"  ⏱️ Running stats after run {i+1}:")
            print(f"     • Average time: {current_avg_time:.2f} seconds")
            print(f"     • Median time: {current_median_time:.2f} seconds")
            if len(execution_times) > 1:
                print(f"     • Std dev: {current_stdev_time:.2f} seconds")
            if memory_usage:
                current_avg_memory = statistics.mean(memory_usage)
                current_median_memory = statistics.median(memory_usage)
                print(f"     • Average memory: {current_avg_memory:.2f} GB")
                print(f"     • Median memory: {current_median_memory:.2f} GB")
            print()
        
        end_total_time = time.time()
        total_time = end_total_time - start_total_time
        
        # Calculate final statistics
        avg_time = statistics.mean(execution_times)
        if len(execution_times) > 1:
            stdev_time = statistics.stdev(execution_times)
            median_time = statistics.median(execution_times)
        else:
            stdev_time = 0
            median_time = execution_times[0]
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        # Memory statistics if available
        avg_memory = statistics.mean(memory_usage) if memory_usage else 0
        median_memory = statistics.median(memory_usage) if memory_usage else 0
        max_memory = max(memory_usage) if memory_usage else 0
        
        # Print detailed benchmark results
        print(f"\n{'='*80}")
        print(f"🔍 FINAL BENCHMARK RESULTS ({benchmark_runs} runs) {mode_emoji}")
        print(f"{'='*80}")
        
        print(f"\n📊 SYSTEM INFO:")
        print(f"  • GPU: {gpu_info}")
        if torch.cuda.is_available():
            print(f"  • GPU Memory: {gpu_memory_gb:.2f} GB")
            print(f"  • CUDA Version: {torch.version.cuda}")
            print(f"  • CUDA Capability: {torch.cuda.get_device_capability(0)}")
        print(f"  • OS: {platform.system()} {platform.release()}")
        print(f"  • Python: {platform.python_version()}")
        print(f"  • PyTorch: {torch.__version__}")
        
        print(f"\n⚙️ CONFIGURATION:")
        print(f"  • Model: {model_info}")
        print(f"  • Mode: {'Image-to-Video' if is_i2v else 'Text-to-Video'}")
        if is_i2v and image_dims:
            print(f"  • Input Image: {image_dims}")
        print(f"  • Resolution: {resolution}")
        print(f"  • Frames: {frames}")
        print(f"  • Fast Mode: {fast_mode}")
        print(f"  • Prompt: {generate_kwargs.get('prompt', 'N/A')}")
        print(f"  • Negative Prompt: {generate_kwargs.get('negative_prompt', 'N/A')}")
        print(f"  • Sample Steps: {generate_kwargs.get('sample_steps', 'N/A')}")
        
        print(f"\n⏱️ RESULTS:")
        print(f"  • Average time: {avg_time:.2f} seconds")
        print(f"  • Median time: {median_time:.2f} seconds")
        print(f"  • Standard deviation: {stdev_time:.2f} seconds")
        print(f"  • Minimum time: {min_time:.2f} seconds")
        print(f"  • Maximum time: {max_time:.2f} seconds")
        print(f"  • Total benchmark time: {total_time:.2f} seconds")
        if memory_usage:
            print(f"  • Average GPU memory usage: {avg_memory:.2f} GB")
            print(f"  • Median GPU memory usage: {median_memory:.2f} GB")
            print(f"  • Maximum GPU memory usage: {max_memory:.2f} GB")
            
        print(f"\n🔢 INDIVIDUAL RUN TIMES:")
        for i, (t, m) in enumerate(zip(execution_times, memory_usage if memory_usage else [0] * len(execution_times))):
            if memory_usage:
                print(f"  • Run {i+1}: {t:.2f} seconds, {m:.2f} GB memory")
            else:
                print(f"  • Run {i+1}: {t:.2f} seconds")
                
        print(f"\n{'='*80}")
        
        # Reset the seed for consistency
        if isinstance(seed, int):
            generate_kwargs['seed'] = seed
        
        return result

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        image: Path | None = None,
        aspect_ratio: str = "16:9",
        frames: int = 81,
        model: str | None = None,
        lora_url: str | None = None,
        lora_strength_model: float = 1.0,
        lora_strength_clip: float = 1.0,
        fast_mode: str = "Balanced",
        sample_shift: float = 8.0,
        sample_guide_scale: float = 5.0,
        sample_steps: int = 30,
        seed: int | None = None,
        resolution: str = "480p",
        replicate_weights: str | None = None,
    ) -> List[Path]:
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        
        # Ensure seed is a proper integer and not a FieldInfo object
        if isinstance(seed, int) or seed is None:
            seed = seed_helper.generate(seed)
        else:
            # If it's a FieldInfo or other object, generate a new random seed
            seed = seed_helper.generate(None)

        if image and model == "1.3b":
            raise ValueError("Image to video generation is not supported for 1.3b")

        image_filename = None
        if image:
            image_filename = self.filename_with_extension(image, "image")
            self.handle_input_file(image, image_filename)

        lora_filename = None
        inferred_model_type = None
        if replicate_weights:
            lora_filename, inferred_model_type = download_replicate_weights(
                replicate_weights, COMFYUI_LORAS_DIR
            )
            model = inferred_model_type
        elif lora_url:
            if m := re.match(
                r"^(?:https?://replicate.com/)?([^/]+)/([^/]+)/?$", lora_url
            ):
                owner, model_name = m.groups()
                lora_filename, inferred_model_type = download_replicate_weights(
                    f"https://replicate.com/{owner}/{model_name}/_weights",
                    COMFYUI_LORAS_DIR,
                )
            elif lora_url.startswith("https://replicate.delivery"):
                lora_filename, inferred_model_type = download_replicate_weights(
                    lora_url, COMFYUI_LORAS_DIR
                )

            if inferred_model_type and inferred_model_type != model:
                print(
                    f"Warning: Model type mismatch between requested model ({model}) and inferred model type ({inferred_model_type}). Using {inferred_model_type}."
                )
                model = inferred_model_type

        if resolution == "720p" and model == "1.3b":
            print("Warning: 720p is not supported for 1.3b, using 480p instead")
            resolution = "480p"

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            fast_mode=fast_mode,
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
            resolution=resolution,
            image_filename=image_filename,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return self.comfyUI.get_files(OUTPUT_DIR, file_extensions=["mp4"])


class StandaloneLoraPredictor(Predictor):
    def predict(
        self,
        prompt: str = Inputs.prompt,
        negative_prompt: str = Inputs.negative_prompt,
        image: Path = Inputs.image,
        aspect_ratio: str = Inputs.aspect_ratio,
        frames: int = Inputs.frames,
        model: str = Inputs.model,
        resolution: str = Inputs.resolution,
        lora_url: str = Inputs.lora_url,
        lora_strength_model: float = Inputs.lora_strength_model,
        lora_strength_clip: float = Inputs.lora_strength_clip,
        fast_mode: str = Inputs.fast_mode,
        sample_steps: int = Inputs.sample_steps,
        sample_guide_scale: float = Inputs.sample_guide_scale,
        sample_shift: float = Inputs.sample_shift,
        seed: int = seed_helper.predict_seed(),
        benchmark: bool = Inputs.benchmark,
        benchmark_runs: int = Inputs.benchmark_runs,
    ) -> List[Path]:
        generate_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "aspect_ratio": aspect_ratio,
            "frames": frames,
            "model": model,
            "resolution": resolution,
            "lora_url": lora_url,
            "lora_strength_model": lora_strength_model,
            "lora_strength_clip": lora_strength_clip,
            "fast_mode": fast_mode,
            "sample_shift": sample_shift,
            "sample_guide_scale": sample_guide_scale,
            "sample_steps": sample_steps,
            "seed": seed,
            "replicate_weights": None,
        }

        if not benchmark:
            return self.generate(**generate_kwargs)
        else:
            # Determine if we're in image-to-video (i2v) or text-to-video (t2v) mode
            mode = "i2v" if image else "t2v"
            model_info = f"{model} ({mode})"
            
            return self.benchmark_generate(
                benchmark_runs=benchmark_runs,
                model_info=model_info,
                resolution=resolution,
                frames=frames,
                fast_mode=fast_mode,
                generate_kwargs=generate_kwargs
            )


class TrainedLoraPredictor(Predictor):
    def predict(
        self,
        prompt: str = Inputs.prompt,
        negative_prompt: str = Inputs.negative_prompt,
        aspect_ratio: str = Inputs.aspect_ratio,
        frames: int = Inputs.frames,
        resolution: str = Inputs.resolution,
        lora_strength_model: float = Inputs.lora_strength_model,
        lora_strength_clip: float = Inputs.lora_strength_clip,
        fast_mode: str = Inputs.fast_mode,
        sample_steps: int = Inputs.sample_steps,
        sample_guide_scale: float = Inputs.sample_guide_scale,
        sample_shift: float = Inputs.sample_shift,
        seed: int = seed_helper.predict_seed(),
        replicate_weights: str = Inputs.replicate_weights,
        benchmark: bool = Inputs.benchmark,
        benchmark_runs: int = Inputs.benchmark_runs,
        image: Path = None,
    ) -> List[Path]:
        generate_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "aspect_ratio": aspect_ratio,
            "frames": frames,
            "model": None,
            "resolution": resolution,
            "lora_url": None,
            "lora_strength_model": lora_strength_model,
            "lora_strength_clip": lora_strength_clip,
            "fast_mode": fast_mode,
            "sample_shift": sample_shift,
            "sample_guide_scale": sample_guide_scale,
            "sample_steps": sample_steps,
            "seed": seed,
            "replicate_weights": replicate_weights,
        }

        if not benchmark:
            return self.generate(**generate_kwargs)
        else:
            # Determine if we're in image-to-video (i2v) or text-to-video (t2v) mode
            mode = "i2v" if image else "t2v"
            model_info = f"LoRA ({mode})" if replicate_weights else f"Default ({mode})"
            
            return self.benchmark_generate(
                benchmark_runs=benchmark_runs,
                model_info=model_info,
                resolution=resolution,
                frames=frames,
                fast_mode=fast_mode,
                generate_kwargs=generate_kwargs
            )
