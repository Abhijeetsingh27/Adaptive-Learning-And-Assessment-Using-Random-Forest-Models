from django.core.management.base import BaseCommand
from core.models import User, Topic, Material, Question

class Command(BaseCommand):
    help = 'Seeds database with initial data'

    def handle(self, *args, **kwargs):
        # Create Admin
        if not User.objects.filter(username='admin').exists():
            User.objects.create_superuser('admin', 'admin@example.com', 'admin123', role='admin')
            self.stdout.write(self.style.SUCCESS('Admin user created (admin/admin123)'))

        # Create Topics
        topics_data = [
            {'name': 'Python Basics', 'difficulty': 'easy', 'desc': 'Learn the fundamentals of Python.'},
            {'name': 'Data Structures', 'difficulty': 'medium', 'desc': 'Lists, Dictionaries, and Sets.'},
            {'name': 'Advanced Algorithms', 'difficulty': 'hard', 'desc': 'Graph theory and dynamic programming.'},
        ]
        
        for t_data in topics_data:
            topic, created = Topic.objects.get_or_create(
                name=t_data['name'],
                defaults={'difficulty': t_data['difficulty'], 'description': t_data['desc']}
            )
            
            # Create Material
            Material.objects.get_or_create(
                topic=topic,
                title=f"Introduction to {t_data['name']}",
                defaults={
                    'content': f"<h1>{t_data['name']}</h1><p>This is the starting material for {t_data['name']}. It covers key concepts...</p>",
                    'level': t_data['difficulty']
                }
            )

            # Create Questions
            for i in range(3):
                Question.objects.get_or_create(
                    topic=topic,
                    question_text=f"What is a key concept in {t_data['name']}?",
                    defaults={
                        'difficulty': t_data['difficulty'],
                        'option_a': 'Concept A',
                        'option_b': 'Concept B',
                        'option_c': 'Concept C',
                        'option_d': 'Concept D',
                        'correct_answer': 'A'
                    }
                )
        
        self.stdout.write(self.style.SUCCESS('Database seeded successfully'))
