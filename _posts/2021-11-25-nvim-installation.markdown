---
layout: post
title:  "Everything about Neovim: Installation, Configuration, and Usage"
date:   2021-11-25 14:00:00 +0900
categories: tool
tags: nvim
author: Phuoc. Pham
---



This aritcle contains all of my complete nvim configuration. Follow the guide there on how to use it. Most of the config below also applies to Ubuntu and Mac (not fully tested). 

Vim is a popular code editor on Unix/Linux systems. Vim is powerful, but it has its own shortcomings as an old editor. To overcome the shortcomings of Vim, preserve its advantages (i.e., compatible with Vim) and make the development of Vim faster, the Neovim project is created.

In this post, I will give a detailed guide on how to install Neovim and configure it as an IDE-like environment for Python development.


#### **Installation**

Firstly, you need to install `nvim` by the following command: `sudo apt-get install -y neovim`

Secondly, make a configuration file as follows:

```bash
mkdir -p ~/.config/nvim/
vi  ~/.config/nvim/init.vim
```

and put the following contents:

```bash
#init.vim

call plug#begin('~/.local/share/nvim/plugged')

Plug 'davidhalter/jedi-vim'
Plug 'Shougo/deoplete.nvim', { 'do': ':UpdateRemotePlugins' }
Plug 'zchee/deoplete-jedi'
Plug 'vim-airline/vim-airline'
Plug 'jiangmiao/auto-pairs'
Plug 'scrooloose/nerdcommenter'
Plug 'sbdchd/neoformat'
Plug 'davidhalter/jedi-vim'
Plug 'scrooloose/nerdtree'
Plug 'neomake/neomake'
Plug 'terryma/vim-multiple-cursors'
Plug 'machakann/vim-highlightedyank'
Plug 'tmhedberg/SimpylFold'
Plug 'morhetz/gruvbox'

call plug#end()

let g:deoplete#enable_at_startup = 1

" NERDCommenter
nmap <C-_> <Plug>NERDCommenterToggle
vmap <C-_> <Plug>NERDCommenterToggle<CR>gv

" Enable alignment
let g:neoformat_basic_format_align = 1

" Enable tab to space conversion
let g:neoformat_basic_format_retab = 1

" Enable trimmming of trailing whitespace
let g:neoformat_basic_format_trim = 1

" disable autocompletion, because we use deoplete for completion
let g:jedi#completions_enabled = 0

" open the go-to function in split, not another buffer
let g:jedi#use_splits_not_buffers = "right"


" NERDTree
" ======================================
let NERDTreeQuitOnOpen=1
let g:NERDTreeMinimalUI=1
nmap <F2> :NERDTreeToggle<CR>

let g:neomake_python_enabled_makers = ['pylint']
call neomake#configure#automake('nrwi', 500)

hi HighlightedyankRegion cterm=reverse gui=reverse
" set highlight duration time to 1000 ms, i.e., 1 second
let g:highlightedyank_highlight_duration = 1000

colorscheme gruvbox
set background=dark " use dark mode

set notermguicolors


nnoremap tn  :tabnew<Space>
nnoremap tk  :tabnext<CR>
nnoremap tj  :tabprev<CR>
nnoremap th  :tabfirst<CR>
nnoremap tl  :tablast<CR>

```

Finally, install `vim-plug` and other dependencies.

```bash
curl -fLo ~/.local/share/nvim/site/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

# If your system is running Python 2.x, then you need
# install Python 3.x before continuing next steps.

# Install Python3
#sudo add-apt-repository ppa:deadsnakes/ppa
#sudo apt update
#sudo apt install python3.7
#virtualenv -p python3.7 ~/env/python37-env
#source ~/env/python37-env/bin/activate

source ~/path/to/python/environment/bin/activate

pip install --upgrade pip
pip install pynvim jedi yapf pylint
 
```

Once you have installed `nvim` and other dependencies, go to `nvim` and install plugins (type: `:PlugInstall` ) and update the lastest version (type: `:PlugUpdate`)



#### **How to use Neovim & Other Installed Plugins ?**

**Leader Key**

The "Leader Key" is a way of extending the power of VIM's shortcuts by using sequences of keys to perform a command. The default leader key is backslash. Therefore, if you have a map of `<Leader>Q`, you can perform that action by typing \Q. [[ref]](https://stackoverflow.com/questions/1764263/what-is-the-leader-in-a-vimrc-file)

Our `<Leader> Key` is (by default): ` \ `


**Function Explorer**

Jump to the definition of class and method to check their implementation details. 
Move the cursor to the class or method you want to check, then press the various supported shortcut provided by jedi-vim:

- `<leader>d`: go to definition
- `<leader>n`: show the usage of a name in current file
- `<leader>r`: rename the function/class name, the old name will be delete and nvim will enter the insert mode.

**Code Folding** 

- `zo`： Open fold in current cursor position
- `zO`： Open fold and sub-fold in current cursor position recursively
- `zc`： Close the fold in current cursor position
- `zC`： Close the fold and sub-fold in current cursor position recursively

**File/Directory Explorer**

- `F2`: Open the NERDTree File Explorer
- `↑ ↓ → ←`: Select file
- `o` or `Enter`: To open file in the current tab (override)
- `t`: To open file in the new tab.

**Code Comment**
1. Use VISUAL mode (`Ctrl+Shift+V`) to select a code block
2. Use `Ctrl+/` to toggle between comment/uncomment.
