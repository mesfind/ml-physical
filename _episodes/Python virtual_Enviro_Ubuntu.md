# Setting Up a Python virtual environment on Ubuntu 
---

## Method 1: Using venv Module (Recommended)

### 1) Open a Terminal

- Press `Ctrl + Alt + T` or search for `Terminal` in your application launcher.

### 2) Navigate to the Desired Directory

- Use the `cd` command to navigate to the location where you want to create the virtual environment. For example:

`cd Documents/my_project`

### 3) Create the Virtual Environment

- Use the `python3 -m venv` command followed by your desired environment name:

`python3 -m venv my_env`

- This creates a virtual environment directory named `my_env` in the current location.

### 4) Activate the Virtual Environment

- Activate the environment by running the following command (replace my_env accordingly):

`source my_env/bin/activate`

### 5) Install Required Packages

- Once the environment is activated, use pip to install the packages you need for your project:

`pip install <package_name>`

### 6) Deactivate the Environment (Optional):

When you're finished, deactivate the environment by typing:

`deactivate`

- This exits the virtual environment and returns you to your system's default Python environment.

---



