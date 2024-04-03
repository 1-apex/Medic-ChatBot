import re
import nltk
from chatbot_brain import brain_response
import warnings
warnings.filterwarnings("ignore")


class ChatBot:
    # exit responses
    exit_commands = ('quit', 'pause', 'exit', 'goodbye', 'bye', 'later', 'no', 'nope', 'sorry')

    patient_data = {
        'id': '',
        'name': '',
        'gender': '',
        'age': '',
        'insured': False
    }

    # Initial greeting function
    def greet(self):
        self.patient_data['name'] = input("Hey, what is your name ?\n")
        self.chat()

    # Exit chat function
    def exit(self, user_response):
        for command in self.exit_commands:
            if command in user_response.split(' '):
                print("EDoc : Ok bye, have a healthy day!")
                return True
        return False

    # General Chat function
    def chat(self):
        print(f"EDoc : Hi {self.patient_data['name']}, how can I assist you today?")
        flag = True

        while flag:
            user_response = input(f"{self.patient_data['name']} : ")
            user_response = user_response.lower()
            word_tokens = []

            if not self.exit(user_response):
                if user_response == 'thank you' or user_response == 'thanks':
                    print('EDoc : Happy to assist!, do you need more assistance?')
                else:
                    word_tokens = word_tokens + nltk.word_tokenize(user_response)
                    final_words = list(set(word_tokens))
                    print('EDoc :', brain_response(user_response))
            else:
                flag = False


E_doc = ChatBot()
E_doc.greet()
