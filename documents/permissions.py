from rest_framework.permissions import BasePermission


class IsDocumentOwner(BasePermission):
    message = "You do not have access to this document."

    def has_object_permission(self, request, view, obj):
        return obj.owner_id == request.user.id
