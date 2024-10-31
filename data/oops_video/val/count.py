import os

def count_mp4_files(directory):
  return len([file for file in os.listdir(directory) if file.endswith('.mp4')])

if __name__ == "__main__":
  current_directory = os.path.dirname(os.path.abspath(__file__))
  mp4_count = count_mp4_files(current_directory)
  print(f"Number of .mp4 files: {mp4_count}")
