INPUT_NAMES = ["UWB_P",	"UWB_Q", "UWB_R",
               "UWB_AX", "UWB_AY", "UWB_AZ",
               "V_air",	"Elevator",	"Aileron", "Rudder"]

OUTPUT_NAMES = ["alpha", "beta"]

INPUT_LEN = {"alpha": {"term": 2, "elements": 2, "all": 10},
             "beta": {"term": 4, "elements": 5, "all": 10}}

# T: Term
A_T = ["V_square", "UWB_AZ"]
A_E = ["V_air", "UWB_AZ"]

B_T = ["Term0", "Term1", "Term2", "Rudder"]
B_E = ["V_air", "UWB_AY", "UWB_P", "UWB_R", "Rudder"]

X_KEY = {"alpha": {"term": A_T, "elements": A_E, "all": INPUT_NAMES},
         "beta": {"term": B_T, "elements": B_E, "all": INPUT_NAMES}}