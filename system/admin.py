from django.contrib import admin
from .models import News
from django.http import HttpResponse
import csv

class ExportToCSV(admin.ModelAdmin):
    def exportToCSV(self, request, queryset):

        meta = self.model._meta
        field_names = [field.name for field in meta.fields]

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=news_export.csv'.format(meta)
        writer = csv.writer(response)

        writer.writerow(field_names)
        for obj in queryset:
            row = writer.writerow([getattr(obj, field) for field in field_names])

        return response

    exportToCSV.short_description = "Eksport wybranych rekordów"

class NewsAdmin(ExportToCSV, admin.ModelAdmin):
    list_display = ('content', 'result', 'model_choice', 'vector_choice', 'analyzed_date', 'entities')
    search_fields = ['content', 'result']
    actions = ['exportToCSV']
    
admin.site.register(News, NewsAdmin)