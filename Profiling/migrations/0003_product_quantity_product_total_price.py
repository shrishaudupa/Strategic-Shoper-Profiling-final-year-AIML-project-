# Generated by Django 4.2.5 on 2024-03-20 17:13

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("Profiling", "0002_product_delete_homepage"),
    ]

    operations = [
        migrations.AddField(
            model_name="product",
            name="quantity",
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name="product",
            name="total_price",
            field=models.DecimalField(
                blank=True, decimal_places=2, max_digits=10, null=True
            ),
        ),
    ]
