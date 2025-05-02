The data analysis script in this directory is based on testing completed for the Integrated Highway Prototype II (IHP2) system, and that is documented in the Integrated Highway Prototype II System Validation Report (report not yet published). The file name of the data analysis script is "ihp2_data_analysis_script.ipynb", and the script is set up as a Jupyter Notebook file. This data analysis script is specifically intended for ROS1 bags that were collected from the IHP2 Validation Testing that occurred in the summer of 2022.

The two vehicle scenario contained in the script corresponds to the two vehicle platoon rehearsal run used to check system readiness for validation.

The three vehicle scenario contained in the script covers the three vehicle platooning configuration used for validation testing.

**Instructions:**

Step 1: Please confirm you have the necessary packages installed to run this script. To do so, run the script install_dependencies.sh in terminal by navigating to the directory with the script file and running the command below:

sudo bash install_dependencies.sh

Step 2: If needed, decompress the bag files you will be using. To do so, run the script bag_decompress.sh in terminal by navigating to the directory with the script file and running the command below (make sure the bag files are in the same directory as the script):

sudo bash bag_decompress.sh

*Please note that the bag decompression script will save all original rosbag files in the directory "<path_to_script>/origbag".

--------------------------------------------------
**Attribution (alphabetical order):**

- Andrew Fortier - R&D Engineer (Leidos)
- Andy Gaines - R&D Engineer (Leidos)
- Andy Lam - General Engineer (Volpe National Transportation Systems Center)
- Ankur Tyagi - Sr. Data Scientist (Leidos)
- Jon Smet - Software Engineer (Leidos)
