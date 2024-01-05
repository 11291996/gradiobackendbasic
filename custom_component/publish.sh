gradio cc create MyComponent --template SimpleTextbox #this will create a custom component folders #can be provided with a template 
gradio cc show #this will show available templates
#- backend/ <- The python code for your custom component
#- frontend/ <- The javascript code for your custom component
#- demo/ <- A sample app using your custom component. Modify this to develop your component!
#- pyproject.toml <- Used to build the package and specify package metadata.
#enter the folder then 
gradio cc dev #it will start a debug server as well 
#then build the package
gradio cc build #this will create tar.gz file and whl file in dist/ folder
pip install <path to whl file> #this will install the custom component
#publish the component
gradion cc publish #this will publish the component to pip 
pip install name_in_pyproject_toml #this will install the component from pip #the name must be free in pip
