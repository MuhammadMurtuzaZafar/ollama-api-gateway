import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';
import { Sun, Moon, LogOut, Home, Key, BarChart, MessageSquare, Book, Zap } from 'lucide-react';

const Navbar = () => {
  const { user, logout } = useAuth();
  const { isDark, toggleTheme } = useTheme();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <nav className="bg-white dark:bg-gray-800 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-3">
              <Zap className="h-8 w-8 text-primary-600 dark:text-primary-400" />
              <div className="border-l border-gray-300 dark:border-gray-600 pl-3">
                <span className="text-xl font-bold text-gray-900 dark:text-white">
                  Ollama API Gateway
                </span>
              </div>
            </Link>
          </div>

          <div className="flex items-center space-x-4">
            <Link
              to="/docs"
              className="flex items-center space-x-1 text-gray-700 dark:text-gray-200 hover:text-primary-600 dark:hover:text-primary-400"
            >
              <Book className="h-5 w-5" />
              <span>API Docs</span>
            </Link>
            {user && (
              <>
                <Link
                  to={user.role === 'admin' ? '/admin' : '/dashboard'}
                  className="flex items-center space-x-1 text-gray-700 dark:text-gray-200 hover:text-primary-600 dark:hover:text-primary-400"
                >
                  <BarChart className="h-5 w-5" />
                  <span>Dashboard</span>
                </Link>
                <Link
                  to="/playground"
                  className="flex items-center space-x-1 text-gray-700 dark:text-gray-200 hover:text-primary-600 dark:hover:text-primary-400"
                >
                  <MessageSquare className="h-5 w-5" />
                  <span>Playground</span>
                </Link>
                <div className="text-sm text-gray-600 dark:text-gray-300">
                  <span className="font-medium">{user.name}</span>
                  <span className="ml-2 px-2 py-1 bg-primary-100 dark:bg-primary-900 text-primary-800 dark:text-primary-200 rounded-full text-xs">
                    {user.role}
                  </span>
                </div>
              </>
            )}

            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600"
              aria-label="Toggle theme"
            >
              {isDark ? (
                <Sun className="h-5 w-5 text-yellow-500" />
              ) : (
                <Moon className="h-5 w-5 text-gray-700" />
              )}
            </button>

            {user && (
              <button
                onClick={handleLogout}
                className="flex items-center space-x-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
              >
                <LogOut className="h-4 w-4" />
                <span>Logout</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
