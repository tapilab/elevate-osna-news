from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired


class MyForm(FlaskForm):
    class Meta:  # Ignoring CSRF security feature.
        csrf = False

    input_field = StringField(label='input:', id='input_field',
                              validators=[DataRequired()])
    select_field = SelectField(label='method:', id='select_field',
                               choices=[('1', 'text'),
                                        ('2', 'user feature'),
                                        ('3', 'text & user feature')])

    submit = SubmitField('Submit')
