from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone

class User(AbstractUser):
    ROLE_CHOICES = (
        ('student', 'Student'),
        ('admin', 'Admin'),
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='student')
    
    # Profile fields (mostly for students)
    current_level = models.CharField(max_length=10, default='medium', choices=[('easy', 'Easy'), ('medium', 'Medium'), ('hard', 'Hard')])
    engagement_score = models.FloatField(default=0.0)
    learning_state = models.CharField(max_length=20, default='focused')

    def is_student(self):
        return self.role == 'student'
    
    def is_admin(self):
        return self.role == 'admin'

class Topic(models.Model):
    DIFFICULTY_CHOICES = [('easy', 'Easy'), ('medium', 'Medium'), ('hard', 'Hard')]
    name = models.CharField(max_length=100)
    difficulty = models.CharField(max_length=10, choices=DIFFICULTY_CHOICES, default='medium')
    description = models.TextField(blank=True)

    def __str__(self):
        return self.name

class Material(models.Model):
    topic = models.ForeignKey(Topic, on_delete=models.CASCADE, related_name='materials')
    title = models.CharField(max_length=200)
    content = models.TextField()  # HTML or Markdown content
    level = models.CharField(max_length=10, choices=Topic.DIFFICULTY_CHOICES, default='medium')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class Question(models.Model):
    topic = models.ForeignKey(Topic, on_delete=models.CASCADE, related_name='questions')
    difficulty = models.CharField(max_length=10, choices=Topic.DIFFICULTY_CHOICES)
    question_text = models.TextField()
    option_a = models.CharField(max_length=255)
    option_b = models.CharField(max_length=255)
    option_c = models.CharField(max_length=255)
    option_d = models.CharField(max_length=255)
    correct_answer = models.CharField(max_length=1, choices=[('A', 'A'), ('B', 'B'), ('C', 'C'), ('D', 'D')])

    def __str__(self):
        return f"{self.topic.name} - {self.difficulty} - {self.question_text[:50]}"

class QuizAttempt(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='quiz_attempts')
    topic = models.ForeignKey(Topic, on_delete=models.CASCADE)
    score = models.FloatField()
    total_questions = models.IntegerField(default=0)
    correct_count = models.IntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)
    difficulty_at_time = models.CharField(max_length=10)

class BehaviorLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='logs')
    topic = models.ForeignKey(Topic, on_delete=models.SET_NULL, null=True, blank=True)
    activity_type = models.CharField(max_length=50) # 'read_material', 'quiz_start', 'quiz_submit'
    time_spent = models.IntegerField(default=0) # in seconds
    timestamp = models.DateTimeField(auto_now_add=True)
    details = models.JSONField(default=dict, blank=True) # Any extra data like retries, skips

class MLResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ml_results')
    predicted_topic = models.ForeignKey(Topic, on_delete=models.SET_NULL, null=True)
    predicted_level = models.CharField(max_length=10)
    predicted_state = models.CharField(max_length=20)
    confidence = models.FloatField(default=0.0)
    timestamp = models.DateTimeField(auto_now_add=True)

class ModelMetrics(models.Model):
    version = models.CharField(max_length=50)
    accuracy = models.FloatField()
    precision_score = models.FloatField()
    recall_score = models.FloatField()
    rmse = models.FloatField()
    updated_at = models.DateTimeField(auto_now=True)
