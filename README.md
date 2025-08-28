# EdgeHD 2.0 - AI Video/Image Processing Platform

Professional Timeline Editor + AI Processing with Background Removal and Upscaling

## 🚀 Quick Start

### Windows
```bash
# Run installation script
install.bat

# If installation fails, run troubleshooting
troubleshoot.bat

# Start the application
npm run dev
```

### macOS/Linux
```bash
# Make scripts executable
chmod +x install.sh troubleshoot.sh

# Run installation script
./install.sh

# If installation fails, run troubleshooting
./troubleshoot.sh

# Start the application
npm run dev
```

## 📋 System Requirements

- **Node.js**: 18.x or higher
- **Python**: 3.11 or higher
- **Conda**: Miniconda3 or Anaconda (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space minimum
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

## 🔧 Installation Issues

### Common Problems

1. **Conda not found**
   - Run `troubleshoot.bat` (Windows) or `./troubleshoot.sh` (macOS/Linux)
   - Install Miniconda manually: https://docs.conda.io/en/latest/miniconda.html

2. **PyTorch installation fails**
   - Check internet connection
   - Try running as administrator (Windows) or with sudo (Linux/macOS)
   - Clear pip cache: `pip cache purge`

3. **npm install fails**
   - Clear npm cache: `npm cache clean --force`
   - Try with legacy peer deps: `npm install --legacy-peer-deps`

4. **Permission errors**
   - Windows: Run as administrator
   - macOS/Linux: Use `sudo` or fix directory permissions

### Manual Installation Steps

If automatic installation fails:

1. **Install Node.js**
   ```bash
   # Download from https://nodejs.org/
   # Or use package manager
   ```

2. **Install Conda**
   ```bash
   # Download Miniconda from https://docs.conda.io/en/latest/miniconda.html
   # Or use package manager
   ```

3. **Create Python environment**
   ```bash
   conda create -n edgehd python=3.11 -y
   conda activate edgehd
   ```

4. **Install PyTorch**
   ```bash
   # CPU version
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
   
   # CUDA version (if you have NVIDIA GPU)
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
   ```

5. **Install backend dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

6. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

7. **Install root dependencies**
   ```bash
   npm install
   ```

## 🏗️ Architecture

- **Backend**: Flask API server (Python 3.11 + PyTorch 2.1.0)
- **Frontend**: Next.js + shadcn/ui (Node.js + React)
- **Timeline**: Professional video editing interface
- **Database**: File-based storage
- **AI Models**: BiRefNet + Real-ESRGAN

## 🎯 Key Features

- **Professional Timeline Editor** with frame-level precision
- **Draggable playhead** for instant navigation
- **Multi-track video/audio editing**
- **AI-powered background removal** and upscaling
- **Responsive design** with dynamic layout

## 🚀 Usage

### Development
```bash
npm run dev          # Start both servers
npm run dev:backend  # Start backend only (http://localhost:8080)
npm run dev:frontend # Start frontend only (http://localhost:3000)
```

### Production
```bash
npm run build        # Build frontend for production
npm run start        # Start both servers (production)
```

### Access URLs
- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8080

## 🔍 Troubleshooting

### Run Diagnostics
```bash
# Windows
troubleshoot.bat

# macOS/Linux
./troubleshoot.sh
```

### Manual Checks

1. **Check Node.js**
   ```bash
   node --version
   npm --version
   ```

2. **Check Python/Conda**
   ```bash
   python --version
   conda --version
   ```

3. **Check network connectivity**
   ```bash
   ping 8.8.8.8
   npm ping
   ```

4. **Check disk space**
   ```bash
   # Windows
   dir /-c
   
   # macOS/Linux
   df -h
   ```

### Reset Installation

If you need to start fresh:

1. **Remove existing installations**
   ```bash
   # Remove conda environment
   conda env remove -n edgehd -y
   
   # Remove node_modules
   rm -rf frontend/node_modules node_modules
   
   # Clear caches
   npm cache clean --force
   pip cache purge
   ```

2. **Reinstall**
   ```bash
   # Run installation script again
   ./install.sh  # or install.bat on Windows
   ```

## 📁 Project Structure

```
EdgeHD/
├── backend/                 # Flask API server
│   ├── app.py              # Main Flask application
│   ├── requirements.txt    # Python dependencies
│   ├── modules/            # AI processing modules
│   └── models/             # AI model storage
├── frontend/               # Next.js frontend
│   ├── src/
│   │   ├── app/           # Next.js app directory
│   │   ├── components/    # React components
│   │   └── hooks/         # Custom React hooks
│   └── package.json       # Node.js dependencies
├── install.bat            # Windows installation script
├── install.sh             # macOS/Linux installation script
├── troubleshoot.bat       # Windows troubleshooting script
├── troubleshoot.sh        # macOS/Linux troubleshooting script
└── package.json           # Root dependencies
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Run the troubleshooting script
3. Check the error messages carefully
4. Ensure all system requirements are met
5. Try manual installation steps

For additional help, please open an issue on GitHub with:
- Your operating system and version
- Node.js and Python versions
- Complete error messages
- Steps to reproduce the issue
