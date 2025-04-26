import torch
import torchvision
import torchvision.transforms as transforms
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import random
import numpy as np
from tqdm import tqdm
import time
import os
from datetime import datetime
import csv
import argparse

# Configuration constants
CONSOLE_WIDTH = 150
CSV_FLUSH_FREQUENCY = 100  # Write to CSV after this many images
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]


def tensor_to_pil_image(tensor_image):
    """Convert a normalized tensor image back to PIL Image format."""
    img = tensor_image.cpu().numpy().transpose((1, 2, 0))
    img = ((img * NORMALIZATION_STD + NORMALIZATION_MEAN) * 255).astype(np.uint8)
    return Image.fromarray(img)


def sanitize_text(text):
    """Sanitize text content for CSV storage by removing newlines and normalizing whitespace."""
    if text is None:
        return ""
    return ' '.join(str(text).replace('\n', ' ').split())


def get_resume_position(csv_path):
    """Determine the last successfully processed image index from an existing CSV file."""
    if not os.path.exists(csv_path):
        return -1
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            # Skip header row
            next(file)
            # Read all lines and find the last valid entry
            lines = file.readlines()
            if not lines:
                return -1
            last_line = lines[-1].strip()
            if not last_line:
                return -1
            # Extract image index (first column)
            last_image_index = int(last_line.split(',')[0]) - 1  # Convert to 0-based index
            return last_image_index
    except Exception as err:
        print(f"Warning: Error reading resume position: {err}")
        return -1


def parse_cli_arguments():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description='CIFAR10 Image Analysis with Vision-Language Model')
    parser.add_argument('--resume', type=str, help='Path to existing CSV file to resume processing from')
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Directory for saving output files (default: output)')
    parser.add_argument('--model', type=str, default='HuggingFaceTB/SmolVLM2-2.2B-Instruct',
                      help='Vision-language model to use for analysis')
    parser.add_argument('--questions-per-image', type=int, default=5,
                      help='Number of random questions to ask per image (default: 5)')
    return parser.parse_args()


class ImageAnalyzer:
    """Main class for analyzing CIFAR10 images with a vision-language model."""
    
    def __init__(self, args):
        self.args = args
        self.output_file = self._setup_output_file()
        self.model_name = args.model
        self.questions_per_image = args.questions_per_image
        self.processor = None
        self.model = None
        self.testset = None
        self.questions_bank = self._prepare_questions_bank()
        
    def _setup_output_file(self):
        """Setup output file path, creating directory if needed."""
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        if self.args.resume:
            if not os.path.exists(self.args.resume):
                raise FileNotFoundError(f"Resume file not found: {self.args.resume}")
            output_file = self.args.resume
            print(f"Resuming analysis from: {output_file}")
        else:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.args.output_dir, f"cifar_analysis_{timestamp}.csv")
            print(f"Starting new analysis, output will be saved to: {output_file}")
        
        return output_file
        
    def _prepare_questions_bank(self):
        """Prepare the bank of questions to ask about each image."""
        return [
            # Object Identification
            "What is the main object in this image?",
            "How many distinct objects can you identify in the image?",
            "Are there any secondary objects in the background?",
            "What is the approximate size of the main object?",
            "Is the main object complete or partially visible?",
            "Can you identify any brand names or logos?",
            "Are there any text or numbers visible in the image?",
            "What type of vehicle or transportation is shown, if any?",
            "Are there any animals or living creatures in the image?",
            "Can you identify any technological devices or equipment?",

            # Color and Lighting
            "What colors are present in the image?",
            "What is the dominant color in this image?",
            "How would you describe the lighting conditions?",
            "Are there any shadows visible in the image?",
            "Is this image bright or dark overall?",
            "Are there any reflective surfaces in the image?",
            "How would you describe the contrast in this image?",
            "Are the colors vibrant or muted?",
            "Is there any color gradient visible?",
            "How does the lighting affect the mood of the image?",

            # Composition and Layout
            "Describe the background of the image.",
            "Is the main subject centered in the frame?",
            "How would you describe the composition of this image?",
            "Is there a clear foreground and background separation?",
            "What perspective is this image taken from?",
            "Is the image symmetrical or asymmetrical?",
            "How much of the frame does the main subject occupy?",
            "Are there any leading lines in the composition?",
            "Is the image cluttered or minimalist?",
            "How would you describe the depth of field?",

            # Texture and Pattern
            "What textures can you identify in the image?",
            "Are there any recurring patterns visible?",
            "How would you describe the surface texture of the main object?",
            "Are there any natural or artificial patterns?",
            "Is there any visible grain or noise in the image?",
            "Can you identify any geometric shapes or patterns?",
            "Are there any interesting textural contrasts?",
            "How does texture contribute to the image's depth?",
            "Are there any smooth or rough surfaces visible?",
            "Can you identify any architectural patterns?",

            # Motion and Action
            "Is there any sense of movement in the image?",
            "Does the image appear to be in motion or static?",
            "Are there any action elements in the scene?",
            "How is motion portrayed in this image, if at all?",
            "Is there any implied direction of movement?",
            "Are there any motion blur effects?",
            "Does the composition suggest movement?",
            "Are there any frozen action moments?",
            "How does the image capture dynamic elements?",
            "Is there any suggestion of past or future movement?",

            # Environment and Context
            "What type of environment is shown in the image?",
            "Is this an indoor or outdoor scene?",
            "What time of day does this appear to be?",
            "What season is suggested by the image?",
            "Are there any weather conditions visible?",
            "What is the general setting of this image?",
            "Can you identify the location type?",
            "Are there any environmental elements visible?",
            "How would you describe the atmosphere?",
            "What context clues are present in the image?",

            # Technical Aspects
            "How would you rate the image quality?",
            "Is the image in focus or blurry?",
            "Are there any visible artifacts or distortions?",
            "How would you describe the resolution?",
            "Is there any visible noise or grain?",
            "How is the white balance in this image?",
            "Are there any lens effects visible?",
            "How would you describe the sharpness?",
            "Are there any exposure issues?",
            "Can you identify any post-processing effects?",

            # Emotional and Aesthetic
            "What mood does this image convey?",
            "How does this image make you feel?",
            "What is the overall aesthetic of the image?",
            "Is there any emotional content visible?",
            "What story does this image tell?",
            "Are there any symbolic elements?",
            "What is the intended message, if any?",
            "How would you describe the visual impact?",
            "What artistic elements are present?",
            "Does the image evoke any particular emotions?",

            # Unique Features
            "What is unique about this image?",
            "Are there any unusual elements present?",
            "What makes this image stand out?",
            "Are there any unexpected details?",
            "What is the most interesting aspect?",
            "Are there any rare or uncommon features?",
            "What catches your attention first?",
            "Is there anything surprising in the image?",
            "What distinguishes this image from others?",
            "Are there any special characteristics?",

            # Details and Specifics
            "What small details are visible in the image?",
            "Can you describe any intricate patterns?",
            "Are there any hidden elements?",
            "What subtle features can you identify?",
            "Are there any fine details worth noting?",
            "Can you spot any minute textures?",
            "What precise characteristics stand out?",
            "Are there any microscopic details visible?",
            "What specific features define this image?",
            "Can you identify any particular markings?"
        ]
    
    def load_dataset(self):
        """Load CIFAR10 test dataset."""
        print("Step 1: Loading CIFAR10 test dataset...")
        print("-" * CONSOLE_WIDTH)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)
        ])
        
        self.testset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=False,
            download=True, 
            transform=transform
        )
        
        print(f"✓ Test dataset loaded successfully! ({len(self.testset)} images available)")
        print(f"✓ Available classes: {', '.join(self.testset.classes)}\n")
    
    def load_model(self):
        """Load vision-language model and processor."""
        print("Step 2: Loading vision-language model and processor...")
        print("-" * CONSOLE_WIDTH)
        
        try:
            print(f"• Loading processor from {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            print("✓ Processor loaded successfully!")
            
            print(f"• Loading model from {self.model_name}")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            print(f"✓ Model loaded successfully! (Using device: {self.model.device})\n")
        except Exception as e:
            print(f"\n❌ Error loading model: {str(e)}")
            raise
    
    def prepare_analysis(self):
        """Prepare for image analysis."""
        print("Step 3: Preparing image analysis...")
        print("-" * CONSOLE_WIDTH)
        
        print(f"• Total available questions: {len(self.questions_bank)}")
        print(f"• Will randomly select {self.questions_per_image} questions per image")
        total_ops = len(self.testset) * self.questions_per_image
        print(f"• Total operations: {total_ops} image-question pairs")
        est_time_min = (total_ops * 4.0) / 60  # Assuming 4 seconds per question
        print(f"• Estimated time: {est_time_min:4.0f} minutes (at 4.0s per question)")
        print(f"• Data will be flushed to CSV every {CSV_FLUSH_FREQUENCY} images\n")
    
    def analyze_image(self, image, questions):
        """Analyze a single image with multiple questions."""
        results = []
        
        for question in questions:
            try:
                # Format input for the model
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": image},
                            {"type": "text", "text": question},
                        ]
                    },
                ]
                
                # Process inputs
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.model.device, dtype=torch.bfloat16)
                
                # Generate response
                generated_ids = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=64
                )
                
                # Decode response
                response = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
                
                # Clean response
                response = sanitize_text(response)
                response = response.replace("User:", "").replace("Assistant:", "").strip()
                response = response.replace(question, "").strip()
                
                results.append((question, response))
                
            except Exception as e:
                results.append((question, f"ERROR: {str(e)}"))
                
        return results
    
    def run_analysis(self):
        """Run the complete image analysis pipeline."""
        print("Step 4: Starting image analysis...")
        print("-" * CONSOLE_WIDTH)
        
        # Starting metrics
        start_time = time.time()
        successful_analyses = 0
        failed_analyses = 0
        
        # Check if resuming from previous run
        start_idx = get_resume_position(self.output_file) + 1
        
        if start_idx > 0:
            print(f"Resuming from image {start_idx + 1} ({start_idx} images already processed)")
            print(f"Using existing file: {self.output_file}")
        else:
            print(f"Starting new analysis")
            print(f"Creating new file: {self.output_file}")
        
        # Show processing information
        remaining_images = len(self.testset) - start_idx
        est_time_min = (remaining_images * self.questions_per_image * 4.0) / 60
        print(f"\nProcessing Information:")
        print(f"• Images remaining    : {remaining_images:,}")
        print(f"• Questions per image : {self.questions_per_image}")
        print(f"• Total operations    : {remaining_images * self.questions_per_image:,}")
        print(f"• Estimated time      : {est_time_min:4.0f} minutes")
        print(f"• Auto-save interval  : Every {CSV_FLUSH_FREQUENCY} images")
        print("-" * CONSOLE_WIDTH)
        
        # Prepare CSV headers
        headers = ['Image_Number', 'Dataset_Index', 'Class_Label']
        for i in range(1, self.questions_per_image + 1):
            headers.extend([f'Q{i}', f'A{i}'])
        headers.extend(['Concat_Q', 'Concat_A'])
        
        # Open file in appropriate mode
        file_mode = 'a' if start_idx > 0 else 'w'
        with open(self.output_file, file_mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if file_mode == 'w':
                writer.writerow(headers)
            
            # Buffer for batch writing
            rows_buffer = []
            
            # Process each image
            for idx in tqdm(range(start_idx, len(self.testset)), 
                          initial=start_idx, total=len(self.testset),
                          desc="Processing images", unit="image"):
                
                # Get image and convert to PIL
                image_tensor, label = self.testset[idx]
                image = tensor_to_pil_image(image_tensor)
                
                # Select random questions
                selected_questions = random.sample(self.questions_bank, self.questions_per_image)
                
                # Prepare row data
                row_data = [
                    f"{idx + 1:05d}",  # Image Number (padded to 5 digits)
                    str(idx),          # Dataset Index
                    self.testset.classes[label]  # Class Label
                ]
                
                # Analyze image
                question_success = 0
                questions_list = []
                answers_list = []
                
                results = self.analyze_image(image, selected_questions)
                
                for qnum, (question, response) in enumerate(results, 1):
                    clean_q = sanitize_text(question)
                    
                    if not response.startswith("ERROR:"):
                        questions_list.append(f'"{clean_q}"')
                        answers_list.append(f'"{response}"')
                        row_data.extend([clean_q, response])
                        question_success += 1
                    else:
                        questions_list.append(f'"{clean_q}"')
                        answers_list.append(f'"ERROR"')
                        row_data.extend([clean_q, response])
                        print(f"\n⚠️  Error on image {idx + 1}, Q{qnum}: {question[:30]}...")
                        print(f"   Error message: {response}")
                        print(f"   Continuing with next question...")
                
                # Add concatenated columns
                row_data.append(", ".join(questions_list))
                row_data.append(", ".join(answers_list))
                
                # Add row to buffer
                rows_buffer.append(row_data)
                
                # Update statistics
                if question_success == len(selected_questions):
                    successful_analyses += 1
                else:
                    failed_analyses += 1
                
                # Flush buffer to CSV file periodically
                if len(rows_buffer) >= CSV_FLUSH_FREQUENCY:
                    writer.writerows(rows_buffer)
                    f.flush()  # Force write to disk
                    rows_buffer = []  # Clear buffer
                    print(f"\n✓ Progress saved at image {idx + 1}")
                
                # Small delay between images
                time.sleep(0.1)
            
            # Write any remaining rows in buffer
            if rows_buffer:
                writer.writerows(rows_buffer)
                f.flush()
        
        # Final statistics
        total_time = time.time() - start_time
        print("\n=== Analysis Complete ===")
        print("-" * CONSOLE_WIDTH)
        print(f"✓ Total images processed: {len(self.testset) - start_idx}")
        print(f"✓ Successful analyses: {successful_analyses}")
        print(f"⚠️  Failed analyses: {failed_analyses}")
        print(f"✓ Total time taken: {total_time:.2f} seconds")
        print(f"✓ Average time per image: {total_time/(len(self.testset) - start_idx):.2f} seconds")
        print(f"\nResults have been saved to: {self.output_file}")
        print("=" * CONSOLE_WIDTH)


def main():
    """Main entry point for the script."""
    print("\n=== CIFAR10 Image Analysis with Vision-Language Model ===")
    print("Starting the analysis process...\n")
    
    try:
        # Parse command line arguments
        args = parse_cli_arguments()
        
        # Initialize analyzer
        analyzer = ImageAnalyzer(args)
        
        # Run the complete pipeline
        analyzer.load_dataset()
        analyzer.load_model()
        analyzer.prepare_analysis()
        analyzer.run_analysis()
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if 'analyzer' in locals() and hasattr(analyzer, 'output_file'):
            print(f"\nTo resume from the last successful save, run:")
            print(f"python {os.path.basename(__file__)} --resume {analyzer.output_file}")


if __name__ == "__main__":
    main() 