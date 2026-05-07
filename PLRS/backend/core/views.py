import requests
import json
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Avg, Count
from django.http import JsonResponse
from django.conf import settings
from .forms import SignUpForm, TopicForm, MaterialForm, QuestionForm
from .models import User, Topic, Material, Question, QuizAttempt, BehaviorLog, MLResult, ModelMetrics
from django.utils import timezone
from django.contrib.auth.forms import AuthenticationForm

ML_API_URL = "http://127.0.0.1:5000"

# --- Helper Functions ---
def call_ml_api(endpoint, data):
    try:
        response = requests.post(f"{ML_API_URL}{endpoint}", json=data, timeout=2)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        return None
    return None

def log_behavior(user, activity_type, details=None, time_spent=0, topic=None):
    BehaviorLog.objects.create(
        user=user,
        topic=topic,
        activity_type=activity_type,
        time_spent=time_spent,
        details=details or {}
    )

# --- Auth Views ---
def landing_page(request):
    if request.user.is_authenticated:
        if request.user.is_student():
            return redirect('student_dashboard')
        elif request.user.is_admin():
            return redirect('admin_dashboard')
    return render(request, 'core/index.html')

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Account created successfully!")
            if user.is_student():
                return redirect('student_dashboard')
            else:
                return redirect('admin_dashboard')
    else:
        form = SignUpForm()
    return render(request, 'core/signup.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            if user.is_student():
                return redirect('student_dashboard')
            else:
                return redirect('admin_dashboard')
    else:
        form = AuthenticationForm()
    return render(request, 'core/login.html', {'form': form})

def user_logout(request):
    logout(request)
    return redirect('landing')

# --- Student Views ---
@login_required
def student_dashboard(request):
    if not request.user.is_student():
        return redirect('admin_dashboard')
    
    # Get recent ML Result
    ml_result = MLResult.objects.filter(user=request.user).order_by('-timestamp').first()
    
    # Get Recommended Topic
    recommended_topic = None
    if ml_result and ml_result.predicted_topic:
        recommended_topic = ml_result.predicted_topic
    else:
        # Fallback: First topic not fully completed or just random
        recommended_topic = Topic.objects.first()

    context = {
        'student': request.user,
        'ml_result': ml_result,
        'recommended_topic': recommended_topic,
        'recent_attempts': QuizAttempt.objects.filter(user=request.user).order_by('-timestamp')[:5]
    }
    return render(request, 'core/student_dashboard.html', context)

@login_required
def topic_list(request):
    topics = Topic.objects.all()
    # Filter by difficulty if needed, or show all
    return render(request, 'core/topic_list.html', {'topics': topics})

@login_required
def read_material(request, material_id):
    material = get_object_or_404(Material, id=material_id)
    
    # Log start
    log_behavior(request.user, 'read_material_start', topic=material.topic)
    
    return render(request, 'core/material.html', {'material': material})

@login_required
def take_quiz(request, topic_id):
    topic = get_object_or_404(Topic, id=topic_id)
    
    if request.method == 'POST':
        # Grade Quiz
        score = 0
        total = 0
        questions = Question.objects.filter(topic=topic, difficulty=request.user.current_level)
        # If no questions at current level, try all fallback
        if not questions.exists():
             questions = Question.objects.filter(topic=topic)
        
        detail_log = {}
        correct_count = 0
        
        for q in questions:
            total += 1
            selected = request.POST.get(f'question_{q.id}')
            if selected == q.correct_answer:
                score += 1
                correct_count += 1
                detail_log[q.id] = 'correct'
            else:
                detail_log[q.id] = 'wrong'
        
        final_score = (score / total) * 100 if total > 0 else 0
        
        # Save Attempt
        attempt = QuizAttempt.objects.create(
            user=request.user,
            topic=topic,
            score=final_score,
            total_questions=total,
            correct_count=correct_count,
            difficulty_at_time=request.user.current_level
        )
        
        # Adaptive Logic
        old_level = request.user.current_level
        if final_score > 80:
            if request.user.current_level == 'easy': request.user.current_level = 'medium'
            elif request.user.current_level == 'medium': request.user.current_level = 'hard'
        elif final_score < 50:
            if request.user.current_level == 'hard': request.user.current_level = 'medium'
            elif request.user.current_level == 'medium': request.user.current_level = 'easy'
        request.user.save()

        # ML Prediction Update
        # Prepare data for ML Service (New Schema: Physiological Simulation)
        # We need to map real quiz metrics to the physiological features the model expects
        # Features: ['HRV', 'Skin_Temperature', 'Expression_Joy', 'Expression_Confusion', 'Steps', 'Session_Duration']
        
        # Heuristic simulation:
        # High accuracy -> High Joy, Low Confusion, Moderate HRV
        # Low accuracy -> Low Joy, High Confusion, Low HRV (stress)
        import random
        
        accuracy = final_score / 100 # Convert to 0-1 scale
        time_spent = 10 * total # Mock time spent based on number of questions
        
        base_joy = accuracy  # 0 to 1
        base_confusion = 1.0 - accuracy # 0 to 1
        
        # Add some noise
        simulated_hrv = 60 + (accuracy * 20) + random.uniform(-5, 5) # Higher HRV generally good
        simulated_skin_temp = 36.5 + random.uniform(-0.5, 0.5)
        simulated_joy = max(0, min(1, base_joy + random.uniform(-0.1, 0.1)))
        simulated_confusion = max(0, min(1, base_confusion + random.uniform(-0.1, 0.1)))
        simulated_steps = 50 + int(attempt.score) # Arbitrary "activity" proxy
        
        ml_payload = {
            'HRV': simulated_hrv,
            'Skin_Temperature': simulated_skin_temp,
            'Expression_Joy': simulated_joy,
            'Expression_Confusion': simulated_confusion,
            'Steps': simulated_steps,
            'Session_Duration': time_spent,
            # Keep original fields just in case model falls back (though our new model won't)
            'topic_accuracy': accuracy,
            'avg_time_per_question': time_spent, # approx
            'retry_rate': 0, 
            'skip_rate': 0, 
            'attempt_count': 1, 
            'engagement_score': (accuracy * 0.5) + 0.5
        }

        # Call ML Service
        ml_response = call_ml_api('/predict', ml_payload)
        
        predicted_state = 'focused' # default
        predicted_score = 0
        
        if ml_response:
            predicted_state = ml_response.get('learning_state', 'focused')
            predicted_score = ml_response.get('next_score_prediction', 0)
            recommended_level = ml_response.get('recommended_level', request.user.current_level)
            
            # Save Result
            MLResult.objects.create(
                user=request.user,
                predicted_topic=topic, 
                predicted_level=recommended_level,
                predicted_state=predicted_state
            )
            # Update user profile based on ML
            request.user.learning_state = predicted_state
            request.user.engagement_score = predicted_score
            
            # Auto-adjust user level based on recommendation ONLY IF it's a downgrade due to confusion
            # This prevents the ML default 'medium' from overwriting legitimate score-based progression
            if recommended_level == 'easy' and predicted_state in ['confusion', 'frustration']:
                if request.user.current_level != 'easy':
                    messages.info(request, f"Adaptive System: Switching level to {recommended_level} based on your {predicted_state} state.")
                    request.user.current_level = 'easy'
            
            request.user.save()
            
        messages.info(request, f"Quiz Complete! Score: {final_score:.1f}%. Level: {old_level} -> {request.user.current_level}")
        return redirect('student_dashboard')

    else:
        # Get questions based on user level
        questions = Question.objects.filter(topic=topic, difficulty=request.user.current_level)
        if not questions.exists():
             questions = Question.objects.filter(topic=topic)
        return render(request, 'core/quiz.html', {'topic': topic, 'questions': questions})

@login_required
def student_profile(request):
    return render(request, 'core/profile.html', {'user': request.user})

@login_required
def student_analytics(request):
    return render(request, 'core/analytics.html')

# --- Admin Views ---
@login_required
def admin_dashboard(request):
    if not request.user.is_admin():
        return redirect('student_dashboard')
    
    stats = {
        'total_students': User.objects.filter(role='student').count(),
        'total_topics': Topic.objects.count(),
        'avg_engagement': User.objects.filter(role='student').aggregate(Avg('engagement_score'))['engagement_score__avg'] or 0,
    }
    ml_metrics = ModelMetrics.objects.last()
    return render(request, 'core/admin_dashboard.html', {'stats': stats, 'metrics': ml_metrics})

@login_required
def manage_topics(request):
    if not request.user.is_admin(): return redirect('student_dashboard')
    topics = Topic.objects.all()
    return render(request, 'core/manage_topics.html', {'topics': topics})

@login_required
def add_topic(request):
    if not request.user.is_admin(): return redirect('student_dashboard')
    if request.method == 'POST':
        form = TopicForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Topic added")
            return redirect('manage_topics')
    else:
        form = TopicForm()
    return render(request, 'core/form_generic.html', {'form': form, 'title': 'Add Topic'})

@login_required
def add_material(request):
    if not request.user.is_admin(): return redirect('student_dashboard')
    if request.method == 'POST':
        form = MaterialForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Material added")
            return redirect('manage_topics')
    else:
        form = MaterialForm()
    return render(request, 'core/form_generic.html', {'form': form, 'title': 'Add Material'})

@login_required
def add_question(request):
    if not request.user.is_admin(): return redirect('student_dashboard')
    if request.method == 'POST':
        form = QuestionForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Question added")
            return redirect('manage_topics')
    else:
        form = QuestionForm()
    return render(request, 'core/form_generic.html', {'form': form, 'title': 'Add Question'})

@login_required
def trigger_retrain(request):
    if not request.user.is_admin(): return redirect('student_dashboard')
    response = call_ml_api('/retrain', {})
    if response:
        # Update Metrics in DB
        metrics = response.get('metrics', {})
        ModelMetrics.objects.create(
            version=str(timezone.now()),
            accuracy=metrics.get('accuracy', 0),
            precision_score=metrics.get('precision', 0),
            recall_score=metrics.get('recall', 0),
            rmse=metrics.get('rmse', 0)
        )
        messages.success(request, "Model retrained successfully.")
    else:
        messages.error(request, "Failed to contact ML Service")
    return redirect('admin_dashboard')

# --- API ---
@login_required
def analytics_data(request):
    # Return JSON for charts
    user = request.user
    attempts = QuizAttempt.objects.filter(user=user).order_by('timestamp')
    data = {
        'labels': [a.topic.name for a in attempts],
        'scores': [a.score for a in attempts],
        'dates': [a.timestamp.strftime('%Y-%m-%d') for a in attempts]
    }
    return JsonResponse(data)
