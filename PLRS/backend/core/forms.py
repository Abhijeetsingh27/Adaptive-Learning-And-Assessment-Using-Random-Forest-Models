from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import User, Topic, Material, Question

class SignUpForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('username', 'email', 'role')

class TopicForm(forms.ModelForm):
    class Meta:
        model = Topic
        fields = ['name', 'difficulty', 'description']

class MaterialForm(forms.ModelForm):
    class Meta:
        model = Material
        fields = ['title', 'content', 'level', 'topic']

class QuestionForm(forms.ModelForm):
    class Meta:
        model = Question
        fields = ['topic', 'difficulty', 'question_text', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_answer']
