import pytest
from pydantic import ValidationError

from backend.auth import NewUser, PasswordChange, ResetPassword


def test_eight_character_password_is_accepted_across_account_flows():
    assert PasswordChange(old_password='old', new_password='12345678').new_password == '12345678'
    assert NewUser(username='engineer1', password='12345678').password == '12345678'
    assert ResetPassword(password='12345678').password == '12345678'
    with pytest.raises(ValidationError):
        PasswordChange(old_password='old', new_password='1234567')
