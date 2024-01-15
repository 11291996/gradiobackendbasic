#If your component supports streaming output, inherit from the StreamingOutput class.
#If you inherit from BlockContext, you also need to set the metaclass to be ComponentMeta. See example below.

from gradio.blocks import BlockContext
from gradio.component_meta import ComponentMeta

set_documentation_group("layout")

@document()
class Row(BlockContext, metaclass=ComponentMeta):
    #methods are defined just like the README.md file mentions
    @abstractmethod
    def preprocess(self, x: Any) -> Any:
        """
        Convert from the web-friendly (typically JSON) value in the frontend to the format expected by the python function.
        """
        return x

    @abstractmethod
    def postprocess(self, y):
        """
        Convert from the data returned by the python function to the web-friendly (typically JSON) value expected by the frontend.
        """
        return y

    def as_example(self, input_data): #ui example
        return next((c[0] for c in self.choices if c[1] == input_data), None) #self.choices is a list of tuples from display_name and value which is defined in the frontend
    
    @abstractmethod
    def api_info(self) -> dict[str, list[str]]: #creating api structure
        """
        A JSON-schema representation of the value that the `preprocess` expects and the `postprocess` returns.
        """
        pass
    
    @abstractmethod #gradio api page's api example
    def example_inputs(self) -> Any:
        """
        The example inputs for this component for API usage. Must be JSON-serializable.
        """
        pass

    @abstractmethod #defining flagging in key_features.py
    def flag(self, x: Any | GradioDataModel, flag_dir: str | Path = "") -> str:
        pass
    
    @abstractmethod #defining reading from flag stored from function above
    def read_from_flag(
        self,
        x: Any,
        flag_dir: str | Path | None = None,
    ) -> GradioDataModel | Any:
        """
        Convert the data from the csv or jsonl file into the component state.
        """
        return x    
    
#defining data model for a component will make it easier to read and modify its code 
#use pydantic library to define data model which is a library that helps auto parses data types 

from gradio.data_classes import FileData, GradioModel, GradioRootModel
from gradio import Component

class VideoData(GradioModel): #defining data model
    video: FileData
    subtitles: Optional[FileData] = None

class Video(Component):
    data_model = VideoData #using data model in component

#gradio root model does not serialize the data to a dictionary.
from typing import List

class Names(GradioModel): #{'names': ['freddy', 'pete']}
    names: List[str]

class NamesRoot(GradioRootModel): #['freddy', 'pete']
    root: List[str]

#to handle files, component must use data model and FileData class
    
#defining events in gradio just like 
from gradio.events import Events
from gradio.components import FormComponent

class MyComponent(FormComponent):

    EVENTS = [ #list of events must be defined like this
        "text_submit",
        "file_upload",
        Events.change
    ]