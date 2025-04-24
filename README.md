Begin by installing the modules in requirements.txt

This backend is designed to be access by a frontend here: https://github.com/shades87/ATS

This backend requires environment variables for gemini and chat GPT, for marking purposes I'm happy to provide them, please send an email from your @curtin email to daniel.m.mcfadyen@gmail.com

This backend is run with the command: python -m uvicorn api:app --reload

Environment Varables should go in a .env file or saved directly to your environment 
(Windows) setx OPEN_API_KEY key
(Windows) setx GEMINI_KEY key

Weights for the ATS model are saved here https://drive.google.com/drive/folders/1Qc92tn9ON8Jr6hCuBaLVB7UOWfUCaFKA?usp=drive_link
create a folder called weights and save them inside the folder
ATS-Wrapper/weights/{Save Weights Here}
