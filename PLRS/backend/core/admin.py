from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, Topic, Material, Question, QuizAttempt, BehaviorLog, MLResult, ModelMetrics

class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'role', 'current_level', 'engagement_score')
    fieldsets = UserAdmin.fieldsets + (
        (None, {'fields': ('role', 'current_level', 'engagement_score', 'learning_state')}),
    )

admin.site.register(User, CustomUserAdmin)
admin.site.register(Topic)
admin.site.register(Material)
admin.site.register(Question)
admin.site.register(QuizAttempt)
admin.site.register(BehaviorLog)
admin.site.register(MLResult)
admin.site.register(ModelMetrics)
