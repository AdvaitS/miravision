# filepath: /home/ubuntu/sam2/services/session_service.py

class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.current_session_id = None
    
    def create_session(self, session_id, session_dir, file_mappings, predictor, 
                      bucket_name, device_name, experiment_name, sample_name, 
                      data_dir, context_dir):
        """
        Create a new session with the given parameters
        """
        self.sessions[session_id] = {
            'session_dir': session_dir,
            'file_mappings': file_mappings,
            'predictor': predictor,
            'inference_state': None,
            'current_frame_idx': None,
            'bucket_name': bucket_name,
            'device_name': device_name,
            'experiment_name': experiment_name,
            'sample_name': sample_name,
            'data_dir': data_dir,
            'context_dir': context_dir
        }
        self.current_session_id = session_id
        return session_id
    
    def get_session(self, session_id):
        """
        Get session data for the given session_id
        """
        return self.sessions.get(session_id)
    
    def session_exists(self, session_id):
        """
        Check if a session with the given ID exists
        """
        return session_id in self.sessions
    
    def set_current_session(self, session_id):
        """
        Set the current active session
        """
        if not self.session_exists(session_id):
            raise ValueError(f"Session {session_id} does not exist")
        self.current_session_id = session_id
    
    def get_current_session(self):
        """
        Get the current active session
        """
        if not self.current_session_id:
            return None
        return self.get_session(self.current_session_id)
    
    def update_session(self, session_id, key, value):
        """
        Update a specific field in the session data
        """
        if not self.session_exists(session_id):
            raise ValueError(f"Session {session_id} does not exist")
        self.sessions[session_id][key] = value
    
    def delete_session(self, session_id):
        """
        Delete a session by ID
        """
        if not self.session_exists(session_id):
            raise ValueError(f"Session {session_id} does not exist")
        
        # Remove from sessions dict
        del self.sessions[session_id]
        
        # Reset current session ID if it was the deleted session
        if self.current_session_id == session_id:
            self.current_session_id = None
            
        return True
    
    def clear_sessions(self):
        """
        Clear all sessions
        """
        self.sessions = {}
        self.current_session_id = None
        return True