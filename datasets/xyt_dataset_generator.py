#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftshift, ifftshift
import math
import matplotlib.pyplot as plt
import gc
import numpy as np
import tifffile
import os
import random
from tqdm import tqdm
from PIL import Image

# =============================================================================
# 座標管理器：處理物理座標和像素座標的轉換
# =============================================================================
class CoordinateManager:
    """統一的座標系統管理"""
    
    def __init__(self, pixel_size, image_size, z_calibration=None):
        """
        初始化座標管理器
        
        Args:
            pixel_size: 像素大小 (米/像素)
            image_size: 圖像尺寸 (像素)
            z_calibration: Z軸校準函數 (可選)
        """
        self.pixel_size = pixel_size
        self.image_size = image_size
        self.center_offset = image_size / 2
        self.z_calibration = z_calibration
        
    def physical_to_pixel(self, coords_physical):
        """
        物理座標 → 像素座標
        
        Args:
            coords_physical: [..., 3] 物理座標 (x, y, z) in meters
            
        Returns:
            coords_pixel: [..., 3] 像素座標 (x_px, y_px, z_px)
        """
        coords_pixel = torch.zeros_like(coords_physical)
        
        # X, Y: 物理距離 → 像素，中心原點 → 左上角原點
        coords_pixel[..., 0] = coords_physical[..., 0] / self.pixel_size + self.center_offset
        coords_pixel[..., 1] = coords_physical[..., 1] / self.pixel_size + self.center_offset
        
        # Z: 物理距離 → 校準Z值
        if self.z_calibration is not None:
            coords_pixel[..., 2] = self.apply_z_calibration(coords_physical[..., 2])
        else:
            # 簡單線性映射：米 → 納米
            coords_pixel[..., 2] = coords_physical[..., 2] * 1e9
        
        return coords_pixel
    
    def pixel_to_physical(self, coords_pixel):
        """像素座標 → 物理座標"""
        coords_physical = torch.zeros_like(coords_pixel)
        
        # X, Y: 像素 → 物理距離，左上角原點 → 中心原點
        coords_physical[..., 0] = (coords_pixel[..., 0] - self.center_offset) * self.pixel_size
        coords_physical[..., 1] = (coords_pixel[..., 1] - self.center_offset) * self.pixel_size
        
        # Z: 校準Z值 → 物理距離
        if self.z_calibration is not None:
            coords_physical[..., 2] = self.inverse_z_calibration(coords_pixel[..., 2])
        else:
            # 簡單線性映射：納米 → 米
            coords_physical[..., 2] = coords_pixel[..., 2] * 1e-9
        
        return coords_physical
    
    def apply_z_calibration(self, z_physical):
        """應用Z軸校準 (需要根據具體校準曲線實現)"""
        # 這裡是示例實現，實際需要根據校準數據
        return z_physical * 1e9  # 簡單轉換為納米
    
    def inverse_z_calibration(self, z_calibrated):
        """Z軸校準的反函數"""
        return z_calibrated * 1e-9  # 簡單轉換回米

# =============================================================================
# 增強版 DifferentiableForwardModel
# =============================================================================
class EnhancedDifferentiableForwardModel(nn.Module):
    """
    增強版可微分正向模型，整合粗糙表面背景處理
    """
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_real = torch.float64
        self.dtype_complex = torch.complex128
        
        # 翻轉控制參數
        self.flip_phasemask_lr = params.get('flip_phasemask_lr', False)
        self.flip_es_ud = params.get('flip_es_ud', False)
        
        # ★ 新增：座標管理器
        self.coord_manager = CoordinateManager(
            pixel_size=params['cam_pixelsize'],
            image_size=params['cam_npixels'],
            z_calibration=params.get('z_calibration', None)
        )
        self.noise_generator = NoiseGenerator(
            gain=params.get('gain', 2.17),
            offset=params.get('offset', 100), 
            read_noise=params.get('read_noise', 2.0)
        )
        # ★ 背景處理配置
        self.background_mode = params.get('background_mode', 'roughness')
        self.base_background = params.get('base_background', 100)
        self.background_scale = params.get('background_scale', 1.0)
        
        # 初始化原有組件
        self._register_constants()
        self._init_coherence_parameters()
        self._init_polarization_parameters()
        
    def _register_constants(self):
        """註冊常數與網格 (保持原有實現並新增像素網格)"""
        p = self.params
        wavelength = torch.tensor(p['wavelength'], dtype=self.dtype_real, device=self.device)
        k = 2 * math.pi / wavelength
        R = ((p['n_glass'] - p['n_medium']) / (p['n_glass'] + p['n_medium']))**2
        T = 1 - R
        ks = p['n_medium'] * k

        self.register_buffer('wavelength', wavelength)
        self.register_buffer('k', k)
        self.register_buffer('R', torch.tensor(R, dtype=self.dtype_real, device=self.device))
        self.register_buffer('T', torch.tensor(T, dtype=self.dtype_real, device=self.device))
        self.register_buffer('ks', ks)

        # 物理座標網格 (原有)
        cam_pixelsize = p['cam_pixelsize']
        cam_npixels = p['cam_npixels']
        coords = torch.linspace(-cam_pixelsize * cam_npixels / 2,
                                cam_pixelsize * cam_npixels / 2 - cam_pixelsize,
                                cam_npixels, dtype=self.dtype_real, device=self.device)
        cam_y, cam_x = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('cam_x', cam_x)  # [H, W] 物理座標
        self.register_buffer('cam_y', cam_y)
        self.register_buffer('cam_r', torch.sqrt(cam_x**2 + cam_y**2))
        self.register_buffer('phi_f', torch.atan2(cam_y, cam_x))

        # ★ 新增：像素座標網格
        pixel_coords = torch.arange(cam_npixels, dtype=self.dtype_real, device=self.device)
        pixel_y, pixel_x = torch.meshgrid(pixel_coords, pixel_coords, indexing='ij')
        self.register_buffer('pixel_x', pixel_x)  # [H, W] 像素座標
        self.register_buffer('pixel_y', pixel_y)

        # 角度離散化
        max_angle = torch.arcsin(torch.tensor(p['NA'] / p['n_oil'], dtype=self.dtype_real, device=self.device))
        num_thetas = p['nThetas']
        thetas = torch.linspace(0, max_angle, num_thetas, dtype=self.dtype_real, device=self.device)
        self.register_buffer('thetas', thetas)
        self.register_buffer('max_angle', max_angle)
        self.register_buffer('d_theta', max_angle / num_thetas)
        self.register_buffer('sin_thetas', torch.sin(thetas))
        self.register_buffer('cos_thetas', torch.cos(thetas))
        
        # 參考光場
        polarization = torch.tensor(p['polarization'], dtype=self.dtype_complex, device=self.device)
        if p['geometry'].lower() == 'iscat':
            E_r = torch.sqrt(self.R) * polarization
        else:
            E_r = torch.sqrt(self.T) * polarization
        self.register_buffer('E_r', E_r * p['attenuation'])
        mu = max_angle / math.pi
        self.register_buffer('mu', mu.clone().detach())


    def apply_flipping_to_phasemask(self, phase_mask):
        """
        對相位遮罩應用左右翻轉

        Args:
            phase_mask: 相位遮罩張量

        Returns:
            翻轉後的相位遮罩
        """
        if self.flip_phasemask_lr:
            return torch.flip(phase_mask, dims=[-1])  # 左右翻轉 (最後一個維度)
        return phase_mask

    def apply_flipping_to_es(self, E_s):
        """
        對散射場應用上下翻轉

        Args:
            E_s: 散射場，可以是張量或字典

        Returns:
            翻轉後的散射場
        """
        if not self.flip_es_ud:
            return E_s

        if isinstance(E_s, dict):
            E_s_flipped = {}
            for key in E_s.keys():
                E_s_flipped[key] = torch.flip(E_s[key], dims=[-3])  # 上下翻轉 (倒數第三個維度，通常是 H)
            return E_s_flipped
        else:
            return torch.flip(E_s, dims=[-3])  # 上下翻轉 (倒數第三個維度，通常是 H)

    def set_flipping_options(self, flip_phasemask_lr=None, flip_es_ud=None):
        """
        動態設置翻轉選項

        Args:
            flip_phasemask_lr: 是否左右翻轉相位遮罩
            flip_es_ud: 是否上下翻轉散射場
        """
        if flip_phasemask_lr is not None:
            self.flip_phasemask_lr = flip_phasemask_lr
            self.params['flip_phasemask_lr'] = flip_phasemask_lr

        if flip_es_ud is not None:
            self.flip_es_ud = flip_es_ud
            self.params['flip_es_ud'] = flip_es_ud

    def _combine_phase_masks(self):
        """組合相位遮罩 (不在這裡應用翻轉)"""
        if not hasattr(self, 'vortex_mask'):
            self._compute_vortex_mask()
        if not hasattr(self, 'zernike_mask'):
            self._compute_zernike_mask()

        # 僅組合，不翻轉
        self.combined_phase_mask = self.vortex_mask * self.zernike_mask
    
    # ---------------------------------------------------------------------------
    # 相干性模組
    # ---------------------------------------------------------------------------
    def _init_coherence_parameters(self):
        """初始化相干性相關參數"""
        p = self.params
        
        # 從參數讀取相干性相關配置
        self.wavelength_bandwidth_nm = torch.tensor(p.get('wavelength_bandwidth_nm', 10), 
                                                 dtype=self.dtype_real, device=self.device)
        self.Lc = torch.tensor(p.get('spatial_coherence_length', 5e-6), 
                             dtype=self.dtype_real, device=self.device)
        self.apply_coherence = p.get('apply_coherence', True)
        
        # 轉換單位與計算相關參數
        self.wavelength_bandwidth_m = self.wavelength_bandwidth_nm * 1e-9
        
        # 根據 van Cittert-Zernike 定理計算相干性參數
        self._calculate_coherence_parameters()
    
    def _calculate_coherence_parameters(self):
        """根據 van Cittert-Zernike 定理計算相干性參數"""
        # 計算時間相干長度
        self.Lt = (self.wavelength**2) / (2 * self.wavelength_bandwidth_m)
        
        # 根據 van Cittert-Zernike 定理，在遠場近似下，
        # 對於給定的相干長度 Lc，有效光源角尺寸 θ 為：θ ≈ λ/Lc
        self.source_angular_size = self.wavelength / self.Lc
        
        # 註冊相干性參數為 buffer
        self.register_buffer('_source_angular_size', self.source_angular_size)
        
        # 計算時間相干因子
        pi_tensor = torch.tensor(math.pi, dtype=self.dtype_real, device=self.device)
        self.gamma_temporal = torch.exp(-pi_tensor * (self.wavelength_bandwidth_m / self.wavelength))
        self.register_buffer('_gamma_temporal', self.gamma_temporal)
    
    def compute_coherence_mask(self, X, Y, delta_l=None):
        """
        計算相干性遮罩，基於 van Cittert-Zernike 定理的遠場近似
        """
        if not self.apply_coherence:
            return torch.ones_like(X)
            
        # 空間相干遮罩
        r = torch.sqrt(X**2 + Y**2)
        spatial_coherence = torch.exp(-(r / self.Lc)**2)
        
        # 時間相干性
        if delta_l is None:
            delta_l = torch.tensor(0.0, dtype=self.dtype_real, device=self.device)
        temporal_coherence = torch.exp(-(delta_l / self.Lt)**2)
        
        # 總相干遮罩
        coherence_mask = spatial_coherence * temporal_coherence * self._gamma_temporal
        
        return coherence_mask
    
    def compute_beam_mask(self, X, Y):
        """計算光束強度分佈遮罩"""
        if not self.apply_coherence:
            return torch.ones_like(X)
            
        # 在遠場中，光束發散與光源角尺寸成反比
        beam_width_parameter = self.wavelength / self.source_angular_size
        
        # 生成具有適當尺寸的光束遮罩
        r = torch.sqrt(X**2 + Y**2)
        beam_mask = torch.exp(-r**2 / beam_width_parameter**2)
        
        return beam_mask

    # ---------------------------------------------------------------------------
    # 偏振調制模組
    # ---------------------------------------------------------------------------
    def _init_polarization_parameters(self):
        """初始化偏振調制相關參數"""
        p = self.params
        self.apply_polar_mod = p.get('apply_polar_mod', False)
        self.qwp_angle = p.get('qwp_angle', math.pi/4)
        
        # 計算Jones矩陣
        self._compute_jones_matrix()
    
    def _compute_jones_matrix(self):
        """計算四分之一波片的Jones矩陣"""
        theta = self.qwp_angle
        
        # 旋轉矩陣元素
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        # 四分之一波片在主軸方向的Jones矩陣: [[1, 0], [0, j]]
        # 經過角度旋轉後的Jones矩陣
        # J = R(-θ) * [[1, 0], [0, j]] * R(θ)
        
        # 直接計算旋轉後的矩陣元素
        j11_real = cos_theta**2 + sin_theta**2 * 0  # cos²θ
        j11_imag = sin_theta**2  # j*sin²θ 的虛部
        
        j12_real = cos_theta * sin_theta * (1 - 0)  # cosθsinθ(1-j) 的實部
        j12_imag = cos_theta * sin_theta * (-1)  # cosθsinθ(1-j) 的虛部
        
        j21_real = cos_theta * sin_theta * (1 - 0)  # cosθsinθ(1-j) 的實部
        j21_imag = cos_theta * sin_theta * (-1)  # cosθsinθ(1-j) 的虛部
        
        j22_real = sin_theta**2 + cos_theta**2 * 0  # sin²θ
        j22_imag = cos_theta**2  # j*cos²θ 的虛部
        
        # 創建Jones矩陣
        jones_real = torch.tensor([
            [j11_real, j12_real],
            [j21_real, j22_real]
        ], dtype=self.dtype_real, device=self.device)
        
        jones_imag = torch.tensor([
            [j11_imag, j12_imag],
            [j21_imag, j22_imag]
        ], dtype=self.dtype_real, device=self.device)
        
        self.jones_matrix = torch.complex(jones_real, jones_imag)
    
    def apply_polarization_modulation(self, E_s):
        """
        對散射場應用偏振調制
        
        Args:
            E_s: 散射場，形狀為 (B, H, W, 2) 或字典
            
        Returns:
            偏振調制後的散射場
        """
        if not self.apply_polar_mod:
            return E_s
        
        if isinstance(E_s, dict):
            E_s_mod = {}
            for key in E_s.keys():
                E_s_mod[key] = self._apply_jones_matrix(E_s[key])
            return E_s_mod
        else:
            return self._apply_jones_matrix(E_s)
    
    def _apply_jones_matrix(self, E_field):
        """應用Jones矩陣到電場"""
        original_shape = E_field.shape
        
        # 重塑為 (N, 2) 形式進行矩陣乘法
        E_flat = E_field.reshape(-1, 2)
        
        # 應用Jones矩陣: (N, 2) @ (2, 2)^T = (N, 2)
        E_mod_flat = torch.matmul(E_flat, self.jones_matrix.T)
        
        # 重塑回原始形狀
        E_mod = E_mod_flat.reshape(original_shape)
        
        return E_mod
    
    def update_qwp_angle(self, new_angle):
        """動態更新QWP角度"""
        self.qwp_angle = new_angle
        self.params['qwp_angle'] = new_angle
        self._compute_jones_matrix()
    
    def set_apply_polar_mod(self, apply_mod):
        """動態設置是否應用偏振調制"""
        self.apply_polar_mod = apply_mod
        self.params['apply_polar_mod'] = apply_mod

    # ---------------------------------------------------------------------------
    # 相位遮罩模組（修改版，支持動態更新）
    # ---------------------------------------------------------------------------
    def _init_phase_masks(self):
        """初始化相位遮罩"""
        self._compute_vortex_mask()
        self._compute_zernike_mask()
        self._combine_phase_masks()
    
    def _compute_vortex_mask(self):
        """計算vortex相位遮罩"""
        p = self.params
        if 'l' in p and p['l'] != 0:
            npix = p['cam_npixels']
            d = p['cam_pixelsize']
            fx = torch.fft.fftshift(torch.fft.fftfreq(npix, d=d)).to(self.device)
            fy = torch.fft.fftshift(torch.fft.fftfreq(npix, d=d)).to(self.device)
            fy2d, fx2d = torch.meshgrid(fy, fx, indexing='ij')
            phi_f = torch.atan2(fy2d, fx2d)
            phase = p['l'] * phi_f
            self.vortex_mask = torch.complex(torch.cos(phase), torch.sin(phase))
        else:
            self.vortex_mask = torch.ones((p['cam_npixels'], p['cam_npixels']),
                                          dtype=self.dtype_complex, device=self.device)
    
    def _compute_zernike_mask(self):
        """計算Zernike相位遮罩"""
        p = self.params
        if 'zernike_coeffs' in p and isinstance(p['zernike_coeffs'], dict) and p['zernike_coeffs'] != 0:
            N = p['cam_npixels']
            x_f = torch.linspace(-1, 1, N, dtype=self.dtype_real, device=self.device)
            y_f = torch.linspace(-1, 1, N, dtype=self.dtype_real, device=self.device)
            X_f, Y_f = torch.meshgrid(x_f, y_f, indexing='ij')
            r_f = torch.sqrt(X_f**2 + Y_f**2)
            r_f = r_f / torch.max(r_f)
            phi_f = torch.atan2(Y_f, X_f)
            
            zernike_mask = torch.ones((N, N), dtype=self.dtype_complex, device=self.device)
            for (m, n), coeff in p['zernike_coeffs'].items():
                Z = r_f**(abs(m)) * torch.cos(n * phi_f)
                phase = 2 * math.pi * coeff * Z
                zernike_mask = zernike_mask * torch.complex(torch.cos(phase), torch.sin(phase))
            self.zernike_mask = zernike_mask
        else:
            self.zernike_mask = torch.ones((p['cam_npixels'], p['cam_npixels']),
                                           dtype=self.dtype_complex, device=self.device)
    
    def _combine_phase_masks(self):
        """組合相位遮罩"""
        if not hasattr(self, 'vortex_mask'):
            self._compute_vortex_mask()
        if not hasattr(self, 'zernike_mask'):
            self._compute_zernike_mask()
        self.combined_phase_mask = self.vortex_mask * self.zernike_mask
    
    def apply_phase_mask(self, E_s):
        """應用相位遮罩到散射場 (修正版，在這裡應用翻轉)"""
        p = self.params
        if ('l' not in p or p['l'] == 0) and            ('zernike_coeffs' not in p or p['zernike_coeffs'] == 0):
            return E_s

        if not hasattr(self, 'combined_phase_mask'):
            self._init_phase_masks()

        # 在使用前應用翻轉
        phase_mask_to_use = self.apply_flipping_to_phasemask(self.combined_phase_mask)

        if isinstance(E_s, dict):
            E_s_mod = {}
            for key in E_s.keys():
                E_s_mod[key] = self._apply_phase_mask_to_field(E_s[key], phase_mask_to_use)
            return E_s_mod
        else:
            return self._apply_phase_mask_to_field(E_s, phase_mask_to_use)
    
    def _apply_phase_mask_to_field(self, E_field, phase_mask=None):
        """對單個電場應用相位遮罩 (修正版，接受外部相位遮罩)"""
        if phase_mask is None:
            phase_mask = self.combined_phase_mask

        E_s_mod = torch.zeros_like(E_field, dtype=self.dtype_complex, device=self.device)
        for pol in range(2):
            E_freq = fftshift(fft2(ifftshift(E_field[..., pol])), dim=(-2, -1))
            E_freq_masked = E_freq * phase_mask
            E_mod = fftshift(ifft2(ifftshift(E_freq_masked)), dim=(-2, -1))
            E_s_mod[..., pol] = E_mod
        return E_s_mod
    
    def update_vortex_parameter(self, l):
        """動態更新vortex參數"""
        self.params['l'] = l
        self._compute_vortex_mask()
        self._combine_phase_masks()
    
    def update_zernike_coeffs(self, zernike_coeffs):
        """動態更新Zernike係數"""
        self.params['zernike_coeffs'] = zernike_coeffs
        self._compute_zernike_mask()
        self._combine_phase_masks()

    # ---------------------------------------------------------------------------
    # 其餘方法保持不變（省略顯示，使用你原來的實現）
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Approximated Bessel functions (可微分版本)
    # ---------------------------------------------------------------------------
    def j0_approx(self, x, threshold=1e-3):
        """
        Pure tensor approximation of Bessel function J0(x) that works on GPU
        Uses series expansion for small x and asymptotic form for large x
        """
        small_mask = (torch.abs(x) < threshold)
        x2 = x**2

        # Series expansion for small x values
        small_val = 1 - x2/4 + x2**2/64 - x2**3/2304 + x2**4/147456

        # For larger values, use this approximation based on asymptotic form
        x_abs = torch.abs(x)
        phase = x_abs - math.pi/4
        large_val = torch.sqrt(2/(math.pi * x_abs)) * torch.cos(phase)

        return torch.where(small_mask, small_val, large_val)

    def j1_approx(self, x, threshold=1e-3):
        """
        Pure tensor approximation of Bessel function J1(x) that works on GPU
        """
        small_mask = (torch.abs(x) < threshold)
        x2 = x**2

        # Series expansion for small x
        small_val = x/2 - (x * x2)/16 + (x * x2**2)/384 - (x * x2**3)/18432

        # Asymptotic approximation for larger x
        x_abs = torch.abs(x)
        phase = x_abs - 3*math.pi/4
        large_val = torch.sqrt(2/(math.pi * x_abs)) * torch.cos(phase)

        # Preserve the sign of x
        large_val = large_val * torch.sign(x)

        return torch.where(small_mask, small_val, large_val)

    def j2_approx(self, x, threshold=1e-3):
        """
        Pure tensor approximation of Bessel function J2(x) that works on GPU
        """
        small_mask = (torch.abs(x) < threshold)
        x2 = x**2

        # Series expansion for small x
        small_val = x2/8 - x2**2/96 + x2**3/3072 - x2**4/184320

        # Asymptotic approximation for larger x
        x_abs = torch.abs(x)
        # Avoid division by zero
        safe_x = torch.where(x_abs < 1e-10, torch.ones_like(x_abs), x_abs)
        phase = x_abs - 5*math.pi/4
        large_val = torch.sqrt(2/(math.pi * safe_x)) * torch.cos(phase)

        return torch.where(small_mask, small_val, large_val)

    # ---------------------------------------------------------------------------
    # 球型 Bessel 函數 (使用遞推計算)
    # ---------------------------------------------------------------------------
    @staticmethod
    def spherical_bessel_j(n, x):
        n = int(n)  # 確保 n 為整數
        eps = 1e-12

        # Handle complex input by manually working with real and imaginary parts
        if x.is_complex():
            x_real = x.real
            x_imag = x.imag
            x_abs2 = x_real**2 + x_imag**2  # |x|²
            small_mask = x_abs2 < eps**2

            # Create safe versions of x that avoid division by very small values
            safe_real = torch.where(small_mask, torch.ones_like(x_real), x_real)
            safe_imag = torch.where(small_mask, torch.zeros_like(x_imag), x_imag)

            # Calculate sin(x)/x for complex x
            # sin(a+bi) = sin(a)cosh(b) + i cos(a)sinh(b)
            sin_real = torch.sin(safe_real) * torch.cosh(safe_imag)
            sin_imag = torch.cos(safe_real) * torch.sinh(safe_imag)

            # Division by complex number z = a+bi: z/w = (a+bi)/(c+di) = (ac+bd)/(c²+d²) + i(bc-ad)/(c²+d²)
            denom = safe_real**2 + safe_imag**2
            res_real = (sin_real * safe_real + sin_imag * safe_imag) / denom
            res_imag = (sin_imag * safe_real - sin_real * safe_imag) / denom

            # For very small x, use limits
            if n == 0:
                # For j₀(x), as x → 0, j₀(x) → 1
                result_real = torch.where(small_mask, torch.ones_like(x_real), res_real)
                result_imag = torch.where(small_mask, torch.zeros_like(x_imag), res_imag)
                return torch.complex(result_real, result_imag)
            else:
                # For higher-order Bessel functions, compute using recurrence relation
                # First compute j₀(x) and j₁(x)
                j0_real = res_real
                j0_imag = res_imag

                # We can avoid the recurrence implementation for now
                # Return jₙ(x) ≈ 0 for n ≥ 1 and small x
                if n >= 1:
                    result_real = torch.where(small_mask, torch.zeros_like(x_real), res_real)
                    result_imag = torch.where(small_mask, torch.zeros_like(x_imag), res_imag)
                    return torch.complex(result_real, result_imag)

                return torch.complex(j0_real, j0_imag)
        else:
            # Original implementation for real inputs
            x_safe = torch.clamp(x, min=eps)
            if n == 0:
                return torch.sin(x_safe) / x_safe
            elif n == 1:
                return torch.sin(x_safe) / (x_safe**2) - torch.cos(x_safe) / x_safe
            else:
                j_nm2 = torch.sin(x_safe) / x_safe  # j0
                j_nm1 = torch.sin(x_safe) / (x_safe**2) - torch.cos(x_safe) / x_safe  # j1
                for nu in range(1, n):
                    j_n = ((2 * nu + 1) / x_safe) * j_nm1 - j_nm2
                    j_nm2, j_nm1 = j_nm1, j_n
                return j_n
    @staticmethod
    def spherical_bessel_y(n, x):
        n = int(n)
        eps = 1e-12

        # Handle complex input
        if x.is_complex():
            # Use absolute value for clamping test and create a mask
            x_abs = torch.abs(x)
            small_mask = x_abs < eps

            # Create a safe version that avoids division by very small values
            safe_x = torch.where(small_mask, torch.ones_like(x), x)

            if n == 0:
                result = -torch.cos(safe_x) / safe_x
                # For small x, y0(x) approaches -infinity, but we'll use a large negative value
                large_neg = torch.ones_like(x) * -1e10
                result = torch.where(small_mask, large_neg, result)
                return result
            elif n == 1:
                result = -torch.cos(safe_x)/(safe_x**2) - torch.sin(safe_x)/safe_x
                # For small x, y1(x) approaches -infinity even faster
                large_neg = torch.ones_like(x) * -1e10
                result = torch.where(small_mask, large_neg, result)
                return result
            else:
                y_nm2 = -torch.cos(safe_x) / safe_x  # y0
                y_nm1 = -torch.cos(safe_x)/(safe_x**2) - torch.sin(safe_x)/safe_x  # y1

                # Handle small x values (both approach -infinity)
                large_neg = torch.ones_like(x) * -1e10
                y_nm2 = torch.where(small_mask, large_neg, y_nm2)
                y_nm1 = torch.where(small_mask, large_neg, y_nm1)

                for nu in range(1, n):
                    y_n = ((2 * nu + 1)/safe_x)*y_nm1 - y_nm2
                    y_nm2, y_nm1 = y_nm1, y_n
                return y_n
        else:
            # Original implementation for real inputs
            x_safe = x.clamp(min=eps)
            if n == 0:
                return -torch.cos(x_safe) / x_safe
            elif n == 1:
                return -torch.cos(x_safe)/(x_safe**2) - torch.sin(x_safe)/x_safe
            else:
                y_nm2 = -torch.cos(x_safe) / x_safe  # y0
                y_nm1 = -torch.cos(x_safe)/(x_safe**2) - torch.sin(x_safe)/x_safe  # y1
                for nu in range(1, n):
                    y_n = ((2 * nu + 1)/x_safe)*y_nm1 - y_nm2
                    y_nm2, y_nm1 = y_nm1, y_n
                return y_n

    # ---------------------------------------------------------------------------
    # 向量化計算 Mie 模型內部項 (避免使用 SciPy)
    # ---------------------------------------------------------------------------
    @staticmethod
    def mie_abcd(m, x, device, dtype):
        nmax = int(torch.round(2 + x + 4 * (x ** (1/3))).item())
        n_range = torch.arange(1, nmax+1, dtype=dtype, device=device)
        z = m * x
        bx = torch.stack([DifferentiableForwardModel.spherical_bessel_j(n.item(), x)/x for n in n_range])
        bz = torch.stack([DifferentiableForwardModel.spherical_bessel_j(n.item(), z)/z for n in n_range])
        yx = torch.stack([DifferentiableForwardModel.spherical_bessel_y(n.item(), x)/x for n in n_range])
        hx = bx + 1j * yx

        b1x_0 = torch.sin(x)/x
        b1x = torch.cat((b1x_0.unsqueeze(0), bx[:-1]), dim=0) if nmax > 1 else b1x_0.unsqueeze(0)
        b1z_0 = torch.sin(z)/z
        b1z = torch.cat((b1z_0.unsqueeze(0), bz[:-1]), dim=0) if nmax > 1 else b1z_0.unsqueeze(0)
        y1x_0 = -torch.cos(x)/x
        y1x = torch.cat((y1x_0.unsqueeze(0), yx[:-1]), dim=0) if nmax > 1 else y1x_0.unsqueeze(0)
        h1x = b1x + 1j * y1x

        n_vec = n_range.to(dtype)
        ax = x * b1x - n_vec * bx
        az = z * b1z - n_vec * bz
        ahx = x * h1x - n_vec * hx

        m2 = m * m
        an = (m2 * bz * ax - bx * az) / (m2 * bz * ahx - hx * az)
        bn = (bz * ax - bx * az) / (bz * ahx - hx * az)
        cn = (bx * ahx - hx * ax) / (bz * ahx - hx * az)
        dn = m * (bx * ahx - hx * ax) / (m2 * bz * ahx - hx * az)
        return an, bn, cn, dn

    @staticmethod
    def mie_pt(u, nmax, device, dtype):
        T = u.shape[0]
        p_val = torch.zeros((nmax, T), dtype=torch.complex128, device=device)
        t_val = torch.zeros((nmax, T), dtype=torch.complex128, device=device)
        p_val[0, :] = 1
        t_val[0, :] = u
        if nmax > 1:
            p_val[1, :] = 3 * u
            t_val[1, :] = 3 * (2*u**2 - 1)
            for n in range(2, nmax):
                p_val[n, :] = ((2 * n - 1)/(n - 1)) * p_val[n-1, :] * u - (n/(n - 1)) * p_val[n-2, :]
                t_val[n, :] = n * u * p_val[n, :] - (n + 1) * p_val[n-1, :]
        return p_val, t_val

    @staticmethod
    def mie_S12(m, x, u, device, dtype):
        an, bn, _, _ = DifferentiableForwardModel.mie_abcd(m, x, device, dtype)
        nmax = an.shape[0]
        p_vals, t_vals = DifferentiableForwardModel.mie_pt(u, nmax, device, dtype)
        n = torch.arange(1, nmax+1, dtype=dtype, device=device).unsqueeze(1)
        n2 = (2*n+1)/(n*(n+1))
        pi_n = p_vals * n2
        tau_n = t_vals * n2
        S1 = torch.sum(an.unsqueeze(1) * pi_n + bn.unsqueeze(1) * tau_n, dim=0)
        S2 = torch.sum(an.unsqueeze(1) * tau_n + bn.unsqueeze(1) * pi_n, dim=0)
        return S1, S2

    # ---------------------------------------------------------------------------
    # 向量化計算散射場：Rayleigh 模型（多粒子批次計算）
    # ---------------------------------------------------------------------------
    def compute_scattered_fields_rayleigh(self, particle_coords):
        """
        向量化計算 Rayleigh 模型下的散射場，支援複數折射率。
        如果粒子的尺寸（D_particle）與折射率（n_particle）相同，則共用參數只計算一次；
        否則對每顆粒子分別計算。
        particle_coords: tensor, shape [B, N, 3]，每個粒子的 (x, y, z)
        回傳: tensor, shape [B, H, W, 2]
        """
        p = self.params
        B, N, _ = particle_coords.shape
        H, W = p['cam_npixels'], p['cam_npixels']

        # 調整粒子座標形狀一次性處理： [B, N, 1, 1]
        x_p = particle_coords[..., 0].view(B, N, 1, 1)
        y_p = particle_coords[..., 1].view(B, N, 1, 1)
        z_p = particle_coords[..., 2].view(B, N, 1, 1)

        # 攝影機網格 (從 buffer 取得，shape [1,1,H,W])
        cam_x = self.cam_x.unsqueeze(0).unsqueeze(0)
        cam_y = self.cam_y.unsqueeze(0).unsqueeze(0)
        # 計算粒子到攝影機網格的距離與角度（只依賴於位置）
        r_d = torch.sqrt((cam_x - x_p)**2 + (cam_y - y_p)**2)
        phi_d = torch.atan2(cam_y - y_p, cam_x - x_p)  # [B, N, H, W]

        n_medium = p['n_medium']
        n_oil = p['n_oil']
        t_oil_ideal = p['t_oil_ideal']
        z_focal = p['z_focal']
        z_camera = p['z_camera']

        # 計算光程相關項：這部分依然按每個粒子位置計算
        t_oil = z_p - z_focal + n_oil * (t_oil_ideal / n_oil - z_p / n_medium)

        # 將 z 和 t_oil 壓縮成 [B, N, 1]，以便與角度項廣播
        z_val = z_p.view(B, N, 1)
        t_val = t_oil.view(B, N, 1)

        # 明確轉換資料類型並確保在同一設備上
        cos_thetas = self.cos_thetas.to(dtype=self.dtype_real, device=self.device).view(1, 1, -1)

        # 計算由粒子高度引起的相位因子，結果 shape 為 [B, N, nThetas]
        aberr_phase = (z_val * n_medium * (cos_thetas + 1) +
                      n_oil * (t_val - t_oil_ideal) * (cos_thetas - 1) -
                      z_camera * cos_thetas - 0.5 * math.pi)

        # 明確轉換成複數類型
        aberr_phase = aberr_phase.to(dtype=self.dtype_real)
        k_val = self.k.to(dtype=self.dtype_real)

        # 使用三角函數計算複數指數，避免直接使用 torch.exp(1j * ...)
        phase_value = k_val * aberr_phase
        abberation_factor = torch.complex(torch.cos(phase_value), torch.sin(phase_value))

        # --- 預計算共用參數部分 ---
        # 檢查 D_particle 與 n_particle 是否為共用參數
        if (not isinstance(p['n_particle'], torch.Tensor) and 
            not isinstance(p['diameter_particle'], torch.Tensor)):

            # 單一計算
            radius = p['diameter_particle'] / 2.0

            # 處理複數折射率的情況
            if isinstance(p['n_particle'], complex):
                # 複數處理
                m_part = complex(p['n_particle'])**2
                m_med = n_medium**2
                alpha = 4 * math.pi * (radius**3) * (m_part - m_med) / (m_part + 2 * m_med)
                # 轉換為 PyTorch 複數
                alpha_tensor = torch.complex(torch.tensor(alpha.real, dtype=self.dtype_real, device=self.device),
                                             torch.tensor(alpha.imag, dtype=self.dtype_real, device=self.device))
            else:
                # 實數處理
                m_part = p['n_particle']**2
                m_med = n_medium**2
                alpha_real = 4 * math.pi * (radius**3) * (m_part - m_med) / (m_part + 2 * m_med)
                alpha_tensor = torch.complex(torch.tensor(alpha_real, dtype=self.dtype_real, device=self.device),
                                            torch.tensor(0.0, dtype=self.dtype_real, device=self.device))

            # 計算 C = (k⁴/6π)|α|²
            alpha_abs_squared = alpha_tensor.real**2 + alpha_tensor.imag**2
            C = (self.ks**4 / (6 * math.pi)) * alpha_abs_squared

            # 計算 E_s0 = μ√C√T·exp(j·∠α)
            amplitude = self.mu * torch.sqrt(C) * torch.sqrt(self.T)
            #phase = torch.angle(alpha_tensor)
            phase = torch.atan2(alpha_tensor.imag, alpha_tensor.real)
            
            E_s0 = amplitude * torch.complex(torch.cos(phase), torch.sin(phase))

            # 擴展 E_s0 到 [B, N] 形狀以便後續廣播
            E_s0 = E_s0.expand(B, N)
        else:
            # 逐個粒子計算
            # 這裡 p['diameter_particle'] 與 p['n_particle'] 為張量或純量
            D = p['diameter_particle']
            n_part = p['n_particle']

            # 確保 D 是張量
            if not isinstance(D, torch.Tensor):
                D = torch.tensor(D, dtype=self.dtype_real, device=self.device).expand(B, N)

            # 處理 n_part 可能是複數的情況
            if isinstance(n_part, complex):
                # 創建複數張量
                n_part_real = torch.tensor(n_part.real, dtype=self.dtype_real, device=self.device).expand(B, N)
                n_part_imag = torch.tensor(n_part.imag, dtype=self.dtype_real, device=self.device).expand(B, N)
                n_part_tensor = torch.complex(n_part_real, n_part_imag)
            elif isinstance(n_part, torch.Tensor):
                # 已經是張量，確保是複數類型
                if n_part.dtype != torch.complex64 and n_part.dtype != torch.complex128:
                    n_part_tensor = torch.complex(n_part, torch.zeros_like(n_part))
                else:
                    n_part_tensor = n_part
            else:
                # 實數，轉換為複數張量
                n_part_tensor = torch.complex(
                    torch.tensor(n_part, dtype=self.dtype_real, device=self.device).expand(B, N),
                    torch.zeros((B, N), dtype=self.dtype_real, device=self.device)
                )

            # 展開到 [B*N]
            D_flat = D.view(-1)
            n_part_flat = n_part_tensor.view(-1)

            # 分離實部和虛部
            n_part_real_flat = n_part_flat.real
            n_part_imag_flat = n_part_flat.imag

            # 合併成屬性對，形狀 [B*N, 3] (D, n_real, n_imag)
            props = torch.stack((D_flat, n_part_real_flat, n_part_imag_flat), dim=1)

            # 求唯一值及逆向索引
            unique_props, inv_idx = torch.unique(props, dim=0, return_inverse=True)

            # 對於每個唯一屬性，計算共用參數
            radius_unique = unique_props[:, 0] / 2.0  # [M]
            n_part_real_unique = unique_props[:, 1]  # [M]
            n_part_imag_unique = unique_props[:, 2]  # [M]

            # 重建複數折射率
            n_part_unique = torch.complex(n_part_real_unique, n_part_imag_unique)  # [M]

            # 計算 m² 
            m_part_unique = n_part_unique**2  # [M]
            m_med = n_medium**2

            # 計算極化率 α
            alpha_unique = 4 * math.pi * (radius_unique**3) * (m_part_unique - m_med) / (m_part_unique + 2 * m_med)  # [M]

            # 計算 C = (k⁴/6π)|α|²
            C_unique = (self.ks**4 / (6 * math.pi)) * torch.abs(alpha_unique)**2  # [M]

            # 計算 E_s0 = μ√C√T·exp(j·∠α)
            amplitude_unique = self.mu * torch.sqrt(C_unique) * torch.sqrt(self.T)
            #phase_unique = torch.angle(alpha_unique)
            phase_unique = torch.atan2(alpha_unique.imag, alpha_unique.real)
            E_s0_unique = amplitude_unique * torch.complex(torch.cos(phase_unique), torch.sin(phase_unique))

            # 將 E_s0_unique 根據 inv_idx 重構為 [B, N]
            E_s0_flat = E_s0_unique[inv_idx]  # [B*N]
            E_s0 = E_s0_flat.view(B, N)  # [B, N]

        # 共用部分（或預計算後的結果）用於後續的角積分
        # Fresnel 係數：與系統參數相關，提前計算
        thetas_g = torch.arcsin(torch.tensor(n_oil, dtype=self.dtype_real, device=self.device) * 
                                self.sin_thetas / p['n_glass'])
        tp = 2 * n_medium * self.cos_thetas / (p['n_glass'] * self.cos_thetas + n_medium * torch.cos(thetas_g))
        ts = 2 * n_medium * self.cos_thetas / (p['n_glass'] * torch.cos(thetas_g) + n_medium * self.cos_thetas)

        # 調整 E_s0、ts 和 tp 的維度以便廣播
        # E_s0 形狀為 [B, N]
        # ts 和 tp 形狀為 [nThetas]
        # 調整為適合廣播的形狀
        E_s0 = E_s0.unsqueeze(-1)  # [B, N, 1]
        ts = ts.unsqueeze(0).unsqueeze(0)  # [1, 1, nThetas]
        tp = tp.unsqueeze(0).unsqueeze(0)  # [1, 1, nThetas]
        cos_thetas_reshaped = cos_thetas  # [1, 1, nThetas]

        # 這部分的結果 shape 為 [B, N, nThetas]
        A0 = E_s0 * (ts + tp * cos_thetas_reshaped)  # [B, N, nThetas]
        A2 = E_s0 * (ts - tp * cos_thetas_reshaped)  # [B, N, nThetas]

        # --- 角度積分部分 ---
        # 調整 sin_thetas 的形狀以便與 bessel_arg 廣播
        sin_thetas_reshaped = self.sin_thetas.view(1, 1, 1, 1, -1)  # [1, 1, 1, 1, nThetas]
        k_val = self.k.to(dtype=self.dtype_real)
        n_oil_tensor = torch.tensor(n_oil, dtype=self.dtype_real, device=self.device)

        # 計算 bessel_arg: [B, N, H, W, nThetas]
        bessel_arg = k_val * n_oil_tensor * r_d.unsqueeze(-1) * sin_thetas_reshaped
        bessel0 = self.j0_approx(bessel_arg)
        bessel2 = self.j2_approx(bessel_arg)

        # 調整 sincos_factor 形狀以便與積分項廣播
        sincos_factor = (self.sin_thetas * torch.sqrt(self.cos_thetas)).view(1, 1, 1, 1, -1)  # [1, 1, 1, 1, nThetas]

        # 調整 abberation_factor 形狀
        abberation_factor_exp = abberation_factor.unsqueeze(2).unsqueeze(2)  # [B, N, 1, 1, nThetas]

        # 重新調整 A0 和 A2 形狀
        A0_exp = A0.unsqueeze(2).unsqueeze(2)  # [B, N, 1, 1, nThetas]
        A2_exp = A2.unsqueeze(2).unsqueeze(2)  # [B, N, 1, 1, nThetas]

        # 計算積分
        I0 = torch.sum(A0_exp * sincos_factor * abberation_factor_exp * bessel0, dim=-1) * self.d_theta  # [B, N, H, W]
        I2 = torch.sum(A2_exp * sincos_factor * abberation_factor_exp * bessel2, dim=-1) * self.d_theta  # [B, N, H, W]

        # 最終計算
        # 使用三角函數而不是直接的複數指數
        # Avoid complex operations by handling real and imaginary parts separately
        prefactor_real = 0.0  # Real part of -j is 0
        prefactor_imag = -k_val / 2.0  # Imaginary part of -j is -1

        # For complex multiplication (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        # When prefactor = 0 + bi, this simplifies to: -b*d + b*c*i
        field_term1 = I0 + I2 * torch.cos(2 * phi_d)  # This is "c" (real part)
        field_term2 = I2 * torch.sin(2 * phi_d)       # This is "d" (real part)

        # Calculate real and imaginary parts separately
        E_s1_real = -prefactor_imag * field_term1.imag
        E_s1_imag = prefactor_imag * field_term1.real
        E_s2_real = -prefactor_imag * field_term2.imag
        E_s2_imag = prefactor_imag * field_term2.real

        # Combine into complex tensors
        E_s1 = torch.complex(E_s1_real, E_s1_imag)
        E_s2 = torch.complex(E_s2_real, E_s2_imag)
        
        E_s_particles = torch.stack((E_s1, E_s2), dim=-1)  # [B, N, H, W, 2]
        E_s_combined = torch.sum(E_s_particles, dim=1)  # [B, H, W, 2]

        return E_s_combined
    # ---------------------------------------------------------------------------
    # 向量化計算散射場：Mie 模型（多粒子批次計算）
    # ---------------------------------------------------------------------------
    def compute_scattered_fields_mie(self, particle_coords):
        p = self.params
        B, N, _ = particle_coords.shape
        H, W = p['cam_npixels'], p['cam_npixels']
        x_p = particle_coords[..., 0].view(B, N, 1, 1)
        y_p = particle_coords[..., 1].view(B, N, 1, 1)
        z_p = particle_coords[..., 2].view(B, N, 1, 1)

        cam_x = self.cam_x.unsqueeze(0).unsqueeze(0)
        cam_y = self.cam_y.unsqueeze(0).unsqueeze(0)
        r_d = torch.sqrt((cam_x - x_p)**2 + (cam_y - y_p)**2)
        phi_d = torch.atan2(cam_y - y_p, cam_x - x_p)

        n_medium = p['n_medium']
        n_oil = p['n_oil']
        t_oil_ideal = p['t_oil_ideal']
        z_focal = p['z_focal']
        z_camera = p['z_camera']
        t_oil = z_p - z_focal + n_oil * (t_oil_ideal / n_oil - z_p / n_medium)
        z_val = z_p.view(B, N, 1)
        t_val = t_oil.view(B, N, 1)
        cos_thetas = self.cos_thetas.view(1, 1, -1)
        aberr_phase = (z_val * n_medium * (cos_thetas - 1) +
                      n_oil * (t_val - t_oil_ideal) * (cos_thetas - 1) -
                      z_camera * cos_thetas)  # [B, N, nThetas]

        # Fix: Using separate real and imaginary components instead of direct complex exponential
        phase_value = self.k * aberr_phase  # This is real tensor
        # Create complex tensor from cos and sin parts
        abberation_factor = torch.complex(torch.cos(phase_value), torch.sin(phase_value))

        radius = p.get('D_particle', 100e-9) / 2.0
        m_rel = p.get('n_particle', 1.45) / n_medium

        # Handle complex refractive index
        if isinstance(m_rel, complex):
            m_rel = torch.tensor(m_rel, dtype=self.dtype_complex, device=self.device)
        else:
            m_rel = torch.tensor(m_rel, dtype=self.dtype_real, device=self.device)

        x_param = self.ks * radius
        mu = self.mu
        amp_prefactor = mu * torch.sqrt(self.T) / (2 * math.pi) * 4 / math.sqrt(6 * math.pi) / n_medium*(4*math.pi)

        thetas_g = torch.arcsin(torch.tensor(n_oil, dtype=self.dtype_real, device=self.device) *
                                  self.sin_thetas / p['n_glass'])
        tp = 2 * n_medium * self.cos_thetas / (p['n_glass'] * self.cos_thetas + n_medium * torch.cos(thetas_g))
        ts = 2 * n_medium * self.cos_thetas / (p['n_glass'] * torch.cos(thetas_g) + n_medium * self.cos_thetas)
        u = -self.cos_thetas  # [nThetas]
        S1, S2 = DifferentiableForwardModel.mie_S12(
            m_rel,
            x_param, u, self.device, self.dtype_real)  # [nThetas]
        S1_full = ts * S1
        S2_full = tp * S2
        A0 = amp_prefactor * (S1_full - S2_full)  # [nThetas]
        A2 = amp_prefactor * (S1_full + S2_full)  # [nThetas]

        bessel_arg = self.k * n_oil * r_d.unsqueeze(-1) * self.sin_thetas  # [B, N, H, W, nThetas]
        bessel0 = self.j0_approx(bessel_arg)
        bessel2 = self.j2_approx(bessel_arg)
        sincos_factor = self.sin_thetas * torch.sqrt(self.cos_thetas)  # [nThetas]

        abberation_factor_exp = abberation_factor.unsqueeze(2).unsqueeze(2)  # [B, N, 1, 1, nThetas]
        I0 = torch.sum(A0 * sincos_factor * abberation_factor_exp * bessel0, dim=-1) * self.d_theta  # [B, N, H, W]
        I2 = torch.sum(A2 * sincos_factor * abberation_factor_exp * bessel2, dim=-1) * self.d_theta  # [B, N, H, W]
        E_s1 = I0 + I2 * torch.cos(2 * phi_d)
        E_s2 = I2 * torch.sin(2 * phi_d)
        E_s_particles = torch.stack((E_s1, E_s2), dim=-1)  # [B, N, H, W, 2]
        E_s_combined = torch.sum(E_s_particles, dim=1)  # [B, H, W, 2]
        return E_s_combined

    # ---------------------------------------------------------------------------
    # 工廠方法：依據 scattering_model 選擇使用哪個模型，並分批計算
    # ---------------------------------------------------------------------------
    def create_scattered_field(self, particle_coords, max_particles_per_batch=None):
        """
        工廠方法：依據 scattering_model 選擇使用哪個模型，並分批計算 (修正版，確保翻轉正確應用)
        """
        model_type = self.params.get('scattering_model', 'rayleigh').lower()

        # 先計算散射場
        if max_particles_per_batch is None:
            if model_type == 'rayleigh':
                E_s = self.compute_scattered_fields_rayleigh(particle_coords)
            elif model_type == 'mie':
                E_s = self.compute_scattered_fields_mie(particle_coords)
            elif model_type == 'both':
                E_s = {
                    'rayleigh': self.compute_scattered_fields_rayleigh(particle_coords),
                    'mie': self.compute_scattered_fields_mie(particle_coords)
                }
            else:
                raise ValueError("Invalid scattering_model. Use 'rayleigh', 'mie', or 'both'.")
        else:
            B, N, _ = particle_coords.shape
            if model_type == 'both':
                E_s_accum_ray = []
                E_s_accum_mie = []
                for start in range(0, N, max_particles_per_batch):
                    end = min(start + max_particles_per_batch, N)
                    sub_coords = particle_coords[:, start:end, :]
                    E_ray = self.compute_scattered_fields_rayleigh(sub_coords)
                    E_mie = self.compute_scattered_fields_mie(sub_coords)
                    E_s_accum_ray.append(E_ray)
                    E_s_accum_mie.append(E_mie)
                E_ray_total = torch.stack(E_s_accum_ray, dim=0).sum(dim=0)
                E_mie_total = torch.stack(E_s_accum_mie, dim=0).sum(dim=0)
                E_s = {'rayleigh': E_ray_total, 'mie': E_mie_total}
            else:
                E_s_accum = []
                for start in range(0, N, max_particles_per_batch):
                    end = min(start + max_particles_per_batch, N)
                    sub_coords = particle_coords[:, start:end, :]
                    if model_type == 'rayleigh':
                        E_sub = self.compute_scattered_fields_rayleigh(sub_coords)
                    elif model_type == 'mie':
                        E_sub = self.compute_scattered_fields_mie(sub_coords)
                    else:
                        raise ValueError(f"Invalid scattering_model: '{model_type}'. Use 'rayleigh', 'mie', or 'both'.")
                    E_s_accum.append(E_sub)
                E_s = torch.stack(E_s_accum, dim=0).sum(dim=0)

        # 對散射場應用上下翻轉
        E_s_flipped = self.apply_flipping_to_es(E_s)

        # 輸出調試信息
        '''
        if self.flip_es_ud:
            print("散射場已進行上下翻轉")
        if self.flip_phasemask_lr:
            print("相位遮罩將進行左右翻轉")
        '''

        return E_s_flipped
    def compute_intensity(self, E_s, E_s_override=None):
        """
        計算強度影像，包含相干性調製（修正版，確保遮罩與場正確對齊）
        """
        p = self.params
        if E_s_override is not None:
            E_s_field = E_s_override
        else:
            E_s_field = E_s
        if isinstance(E_s_field, dict):
            key = list(E_s_field.keys())[0]
            E_s_field = E_s_field[key]

        # 參考光場
        E_r_field = self.E_r.view(1, 1, 2).expand(E_s_field.shape[0], p['cam_npixels'], p['cam_npixels'], 2)

        # 計算強度分量
        Ir = torch.sum(E_r_field.real**2 + E_r_field.imag**2, dim=-1)
        Is = torch.sum(E_s_field.real**2 + E_s_field.imag**2, dim=-1)

        # 根據參數決定是否應用相干性效應
        if self.apply_coherence:
            coherence_mask = self.compute_coherence_mask(self.cam_x, self.cam_y)
            beam_mask = self.compute_beam_mask(self.cam_x, self.cam_y)

            # ★ 關鍵修正：讓遮罩跟著散射場一起翻轉
            if self.flip_es_ud:
                coherence_mask = torch.flip(coherence_mask, dims=[-2])  # y 方向翻轉
                beam_mask = torch.flip(beam_mask, dims=[-2])
        else:
            coherence_mask = torch.ones_like(self.cam_x)
            beam_mask = torch.ones_like(self.cam_x)

        # 計算干涉項並應用相干掩碼
        interference = 2 * torch.real(torch.sum(E_r_field.conj() * E_s_field, dim=-1)) * coherence_mask

        # 組合最終強度，應用光束掩碼
        I = (Ir + Is + interference) * beam_mask

        return I, E_s_field

    def compute_ipsf(self, E_s, particle_coords=None, return_per_particle_info=False):
        """
        計算干涉式點擴散函數 (iPSF)

        Args:
            E_s: 總散射場
            particle_coords: 粒子座標 [B, N, 3]
            return_per_particle_info: 是否返回每個粒子的信息

        Returns:
            如果 return_per_particle_info=False: iPSF
            如果 return_per_particle_info=True: (iPSF, per_particle_info)
                其中 per_particle_info = {
                    'ipsf_singles': [B, N, H, W],  # 每個粒子的單獨iPSF
                    'contrasts': [B, N],            # 每個粒子的對比度
                    'E_s_singles': [B, N, H, W, 2] # 每個粒子的散射場（可選）
                }
        """
        p = self.params
        if isinstance(E_s, dict):
            key = list(E_s.keys())[0]
            E_s = E_s[key]

        # 參考光場
        E_r_field = self.E_r.view(1, 1, 2).expand(E_s.shape[0], p['cam_npixels'], p['cam_npixels'], 2)
        Er_abs2 = torch.sum(E_r_field.real**2 + E_r_field.imag**2, dim=-1) + 1e-20

        # 散射場強度項
        Es_abs2 = torch.sum(E_s.real**2 + E_s.imag**2, dim=-1)

        # 初始化每粒子信息存儲
        if return_per_particle_info and particle_coords is not None:
            B, N, _ = particle_coords.shape
            H, W = p['cam_npixels'], p['cam_npixels']
            per_particle_ipsf = torch.zeros(B, N, H, W, device=self.device, dtype=self.dtype_real)
            per_particle_contrast = torch.zeros(B, N, device=self.device, dtype=self.dtype_real)

        # 根據相干性設置計算干涉項
        if self.apply_coherence and particle_coords is not None:
            # 分解總散射場為每個粒子的貢獻
            B, N, _ = particle_coords.shape
            interference_total = torch.zeros_like(E_r_field[..., 0], dtype=torch.float64)

            for n in range(N):
                x_offset = particle_coords[0, n, 0]
                y_offset = particle_coords[0, n, 1]
                single_particle_coords = particle_coords[:, n:n+1, :].clone()

                with torch.no_grad():
                    E_s_n = self.create_scattered_field(single_particle_coords)
                    if isinstance(E_s_n, dict):
                        key = list(E_s_n.keys())[0]
                        E_s_n = E_s_n[key]

                # 計算粒子相干性遮罩
                particle_coherence_mask = torch.exp(-((self.cam_x - x_offset)**2 + 
                                                     (self.cam_y - y_offset)**2) / self.Lc**2)

                # 關鍵修正：讓粒子遮罩跟著散射場一起翻轉
                if self.flip_es_ud:
                    particle_coherence_mask = torch.flip(particle_coherence_mask, dims=[-2])

                interference_n = 2 * torch.real(torch.sum(E_r_field.conj() * E_s_n, dim=-1))
                interference_total += interference_n * particle_coherence_mask

                # ★ 如果需要返回每粒子信息
                if return_per_particle_info:
                    # 計算單粒子的iPSF (不含參考光強度)
                    Es_n_abs2 = torch.sum(E_s_n.real**2 + E_s_n.imag**2, dim=-1)
                    ipsf_single = (Es_n_abs2 + interference_n) / Er_abs2

                    per_particle_ipsf[:, n] = ipsf_single

                    # 計算對比度
                    contrast = (ipsf_single.max() - ipsf_single.min()).item()
                    if self.params.get('l', 0) != 0:
                        contrast = contrast / 2.0
                    per_particle_contrast[:, n] = contrast

            iPSF = (Es_abs2 + interference_total) / Er_abs2

        else:
            # 簡單方法
            if self.apply_coherence:
                raise ValueError("當 apply_coherence=True 時，必須提供 particle_coords 參數")

            # 不考慮相干性的簡單計算
            interference = 2 * torch.real(torch.sum(E_r_field.conj() * E_s, dim=-1))
            iPSF = (Es_abs2 + interference) / Er_abs2

            # ★ 對於無相干性情況，也計算每粒子信息
            if return_per_particle_info and particle_coords is not None:
                B, N, _ = particle_coords.shape
                for n in range(N):
                    single_particle_coords = particle_coords[:, n:n+1, :].clone()

                    with torch.no_grad():
                        E_s_n = self.create_scattered_field(single_particle_coords)
                        if isinstance(E_s_n, dict):
                            key = list(E_s_n.keys())[0]
                            E_s_n = E_s_n[key]

                    Es_n_abs2 = torch.sum(E_s_n.real**2 + E_s_n.imag**2, dim=-1)
                    interference_n = 2 * torch.real(torch.sum(E_r_field.conj() * E_s_n, dim=-1))
                    ipsf_single = (Es_n_abs2 + interference_n) / Er_abs2

                    per_particle_ipsf[:, n] = ipsf_single

                    contrast = (ipsf_single.max() - ipsf_single.min()).item()
                    if self.params.get('l', 0) != 0:
                        contrast = contrast / 2.0
                    per_particle_contrast[:, n] = contrast

        iPSF_real = torch.real(iPSF)

        if return_per_particle_info and particle_coords is not None:
            per_particle_info = {
                'ipsf_singles': per_particle_ipsf,      # [B, N, H, W]
                'contrasts': per_particle_contrast,      # [B, N]
                'photon_counts': per_particle_contrast * self.params.get('photon_scale', 40000)  # [B, N]
            }
            return iPSF_real, per_particle_info

        return iPSF_real
    
    # ★ 主要接口：完整的仿真流程

    def simulate_with_background(self, particle_coords, roughness_field=None, 
                                add_noise=True, photon_scale=40000, noise_seed=None):
        """修改版：使用 iPSF 作為粒子強度"""
        B = particle_coords.shape[0]

        # 1. 計算粒子散射場
        E_s = self.create_scattered_field(particle_coords)
        E_s_polarized = self.apply_polarization_modulation(E_s)
        E_s_final = self.apply_phase_mask(E_s_polarized)

        # ★ 修改：使用 compute_ipsf 取得粒子強度和每粒子信息
        particle_ipsf, per_particle_info = self.compute_ipsf(
            E_s_final, 
            particle_coords=particle_coords,
            return_per_particle_info=True
        )

        # ★ 粒子強度現在是 iPSF（歸一化的，不含|E_r|²）
        particle_intensity = particle_ipsf

        # ★ 獲取每個粒子的光子數
        photon_counts_per_particle = per_particle_info['contrasts'] * photon_scale

        # 粒子強度偏移到1
        particle_shifted = particle_intensity + 1.0

        # 2. 處理背景 (粗糙度)
        if roughness_field is not None:
            if isinstance(roughness_field, np.ndarray):
                roughness_field = torch.tensor(roughness_field, 
                                             device=self.device, dtype=self.dtype_real)

            # 確保形狀匹配
            if roughness_field.dim() == 2:
                roughness = roughness_field.unsqueeze(0).expand(B, -1, -1)
            else:
                roughness = roughness_field[:B] if len(roughness_field) >= B else roughness_field[0:1].expand(B, -1, -1)

            # ★ 粗糙度偏移到1
            roughness_shifted = roughness + 1.0
        else:
            # 如果沒有粗糙表面，使用均勻背景
            roughness_shifted = torch.ones_like(particle_intensity)

        # 3. ★ 修復：正確的模型應該是背景 + 粒子訊號
        # 因為 roughness_shifted=1+roughness, particle_shifted=1+ipsf
        # 總強度 = 1 + roughness + ipsf = roughness_shifted + particle_shifted - 1.0
        clean_normalized = roughness_shifted + particle_shifted - 1.0
        clean_total = clean_normalized * photon_scale

        # 4. 計算座標資訊
        coords_px = self.coord_manager.physical_to_pixel(particle_coords)

        # 5. 添加實驗性雜訊 (保留原有的 NoiseGenerator)
        if add_noise:
            if noise_seed is not None:
                np.random.seed(noise_seed)

            # 轉換為numpy並使用 NoiseGenerator
            clean_numpy = clean_total.cpu().numpy()
            noisy_scenes = []

            for b in range(B):
                # ★ 使用原有的實驗性雜訊模型
                # 注意：這裡 clean_numpy[b] 已經是光子數了
                noisy_frame, photons = self.noise_generator.add_poisson_experiment(
                    clean_numpy[b], N_scale=1.0, seed=noise_seed  # N_scale=1 因為已經縮放過了
                )
                noisy_scenes.append(photons)

            noisy_scene = torch.tensor(np.array(noisy_scenes), device=self.device, dtype=self.dtype_real)
        else:
            noisy_scene = clean_total

        # 6. ★ 為了保存，計算縮放後的粒子和背景圖像
        particle_scaled = particle_shifted * photon_scale
        background_scaled = roughness_shifted * photon_scale

        # 7. 組織輸出
        result = {
            # 圖像數據
            'noisy_image': noisy_scene,                    # [B, H, W] 帶雜訊的總圖像 (實驗雜訊後)
            'clean_image': clean_total,                    # [B, H, W] 無雜訊的總圖像 (縮放後)
            'particle_image': particle_scaled,             # [B, H, W] 純粒子圖像 (縮放後)
            'background_image': background_scaled,         # [B, H, W] 背景圖像 (縮放後)

            # 未縮放的原始數據（用於檢查）
            'particle_raw': particle_intensity,            # [B, H, W] 原始粒子強度
            'particle_shifted': particle_shifted,          # [B, H, W] 偏移後的粒子
            'roughness_shifted': roughness_shifted,        # [B, H, W] 偏移後的粗糙度

            # 座標數據
            'coords_physical': particle_coords,            # [B, N, 3] 物理座標
            'coords_pixel': coords_px,                     # [B, N, 3] 像素座標
            'particle_z': coords_px[..., 2],              # [B, N] 粒子Z座標（像素/納米）

            # 系統參數
            'z_focal': self.params['z_focal'],            # 標量：焦平面位置（米）
            # ★ 新增：每粒子信息
            'photon_counts_per_particle': photon_counts_per_particle,  # [B, N]
            'contrasts_per_particle': per_particle_info['contrasts'],  # [B, N]
            
            # 元數據
            'metadata': {
                'photon_scale': photon_scale,
                'background_mode': self.background_mode,
                'has_noise': add_noise,
                'noise_seed': noise_seed,
                'noise_params': {
                    'gain': self.noise_generator.gain,
                    'offset': self.noise_generator.offset,
                    'read_noise': self.noise_generator.read_noise
                }
            }
        }

        return result

  
    


# =============================================================================
# 雜訊生成器 (從原有代碼複製)
# =============================================================================
class NoiseGenerator:
    """模擬相機雜訊生成器"""
    
    def __init__(self, gain=1.0, offset=0.0, read_noise=0.0):
        self.gain = gain
        self.offset = offset
        self.read_noise = read_noise
        self.rng = np.random.default_rng()
        
    def add_poisson_experiment(self, img_clean: np.ndarray, N_scale: float, seed: int = None) -> tuple:
        """添加泊松雜訊"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
        mu = img_clean * N_scale
        photons = self.rng.poisson(mu)
        adu = photons * self.gain + self.offset
        
        if self.read_noise > 0:
            adu = adu + self.rng.normal(0.0, self.read_noise, adu.shape)
            
        return adu, photons

# =============================================================================
# 增強版 SimulationPipeline2
# =============================================================================
class EnhancedSimulationPipeline:
    """
    增強版仿真管道，整合粗糙表面背景處理
    """
    
    def __init__(self, optical_params, noise_params=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化光學模型
        self.optical_model = EnhancedDifferentiableForwardModel(optical_params).to(self.device)
        
        # 初始化雜訊生成器
        if noise_params is None:
            noise_params = {'gain': 2.17, 'offset': 100, 'read_noise': 2.0}
        self.noise_generator = NoiseGenerator(**noise_params)
        
        # 保存參數
        self.optical_params = optical_params
        self.noise_params = noise_params
        
        print(f"EnhancedSimulationPipeline 初始化完成，使用設備: {self.device}")

    def generate_rough_surfaces(self, n_surfaces, surface_params, output_dir=None, 
                               output_filename="rough_surfaces", save_files=True):
        """生成粗糙表面 (保持原有實現)"""
        print(f"生成 {n_surfaces} 個粗糙表面...")
        
        N = surface_params['N']
        pixel_size = surface_params['pixel_size']
        sigma = surface_params['sigma']
        xi = surface_params['xi']
        
        surfaces = []
        for i in tqdm(range(n_surfaces), desc="生成粗糙表面"):
            surface = self._generate_single_rough_surface(N, pixel_size, sigma, xi)
            surfaces.append(surface.cpu().numpy())
        
        surfaces_array = np.array(surfaces)
        metadata = surface_params.copy()
        
        if save_files and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            npz_path = os.path.join(output_dir, f"{output_filename}.npz")
            np.savez(npz_path, surfaces=surfaces_array, **metadata)
            print(f"粗糙表面已保存: {npz_path}")
        
        return surfaces_array, metadata

    def _generate_single_rough_surface(self, N, pixel_size, sigma, xi):
        """生成單個粗糙表面"""
        dk = 2 * math.pi / (N * pixel_size)
        k = torch.fft.fftshift(torch.arange(-N//2, N//2, device=self.device) * dk)
        Kx, Ky = torch.meshgrid(k, k, indexing='ij')
        Ksq = Kx**2 + Ky**2
        
        xi_tensor = torch.tensor(xi, device=self.device, dtype=torch.float32)
        PSD = torch.exp(-0.5 * Ksq * xi_tensor**2)
        PSD[0, 0] = 0
        
        noise = torch.randn((N, N), device=self.device, dtype=torch.float32)
        noise_ft = torch.fft.fft2(noise)
        surface = torch.real(torch.fft.ifft2(noise_ft * PSD))
        
        current_std = torch.std(surface)
        if current_std > 1e-10:
            surface = surface * (sigma / current_std)
        else:
            surface = torch.randn((N, N), device=self.device, dtype=torch.float32) * sigma
        
        return surface

    def generate_particle_coordinates(self, n_configs, generation_params):
        """
        生成粒子座標配置
        
        Args:
            n_configs: 配置數量
            generation_params: 生成參數
            
        Returns:
            coords_list: 粒子座標列表
        """
        coords_list = []
        
        for i in range(n_configs):
            if generation_params['type'] == 'random':
                coords = self._generate_random_coordinates(generation_params)
            elif generation_params['type'] == 'single':
                coords = self._generate_single_coordinate(generation_params)
            elif generation_params['type'] == 'grid':
                coords = self._generate_grid_coordinates(generation_params)
            else:
                raise ValueError(f"Unknown coordinate generation type: {generation_params['type']}")
            
            coords_list.append(coords)
        
        return coords_list

    # 1. 修改粒子座標生成函數，支援Z範圍
    def _generate_random_coordinates(self, params):
        """生成隨機粒子座標（支援Z範圍）"""
        n_particles = params['n_particles']
        field_of_view = params['field_of_view']

        # 支援Z範圍 [z_min, z_max]
        z_range = params.get('z_range', [50e-9, 50e-9])  # 默認固定值

        x = torch.rand(1, n_particles, dtype=torch.float64, device=self.device) * 2 * field_of_view - field_of_view
        y = torch.rand(1, n_particles, dtype=torch.float64, device=self.device) * 2 * field_of_view - field_of_view

        # Z隨機分布在範圍內
        z = torch.rand(1, n_particles, dtype=torch.float64, device=self.device) * (z_range[1] - z_range[0]) + z_range[0]

        return torch.stack((x, y, z), dim=2) # [1, n_particles, 3]

 

    def _generate_single_coordinate(self, params):
        """生成單個粒子座標"""
        x_pos = params.get('x_position', 0)
        y_pos = params.get('y_position', 0)
        z_pos = params.get('z_position', 50e-9)
        
        x = torch.ones(1, 1, dtype=torch.float64, device=self.device) * x_pos
        y = torch.ones(1, 1, dtype=torch.float64, device=self.device) * y_pos
        z = torch.ones(1, 1, dtype=torch.float64, device=self.device) * z_pos
        
        return torch.stack((x, y, z), dim=2) # [1, 1, 3]
    
    def _generate_grid_coordinates(self, params):
        """生成網格排列的粒子座標"""
        grid_size = params.get('grid_size', 3)  # 3x3網格
        spacing = params.get('spacing', 1e-6)   # 間距
        z_position = params.get('z_position', 50e-9)
        
        # 創建網格
        coords_1d = torch.linspace(-spacing * (grid_size-1)/2, 
                                  spacing * (grid_size-1)/2, 
                                  grid_size, device=self.device, dtype=torch.float64)
        
        y_grid, x_grid = torch.meshgrid(coords_1d, coords_1d, indexing='ij')
        
        n_particles = grid_size * grid_size
        y_grid, x_grid = torch.meshgrid(coords_1d, coords_1d, indexing='ij')
        
        n_particles = grid_size * grid_size
        x = x_grid.reshape(1, n_particles)
        y = y_grid.reshape(1, n_particles)  
        z = torch.ones(1, n_particles, dtype=torch.float64, device=self.device) * z_position
        
        return torch.stack((x, y, z), dim=2) # [1, N, 3]

    def estimate_photon_counts(self, particle_coords, base_intensity=1000):
        """
        ★ 正確的光子數估算：簡單的+1偏移
        
        Args:
            particle_coords: [B, N, 3] 粒子座標
            base_intensity: 基礎強度
            
        Returns:
            photon_counts: [B, N] 光子數
        """
        B, N, _ = particle_coords.shape
        
        # 簡單的+1偏移 (如您建議)
        photon_counts = torch.ones(B, N, device=self.device) * base_intensity + 1
        
        return photon_counts
   # 2. 修改光學參數處理，支援動態參數
    def process_dynamic_params(self, base_params, dynamic_ranges=None):
            """
            處理動態參數範圍

            Args:
                base_params: 基礎參數字典
                dynamic_ranges: 動態範圍字典，例如：
                    {
                        'D_particle': [10e-9, 20e-9],
                        'n_particle': [1.4, 1.5],
                        'z_focal': [40e-9, 60e-9]
                    }

            Returns:
                params: 處理後的參數（隨機採樣）
            """
            params = base_params.copy()

            if dynamic_ranges:
                for key, value_range in dynamic_ranges.items():
                    if isinstance(value_range, list) and len(value_range) == 2:
                        # 對於複數折射率，分別處理實部和虛部
                        if key == 'n_particle' and isinstance(base_params.get(key), complex):
                            # 假設範圍格式為 [[real_min, real_max], [imag_min, imag_max]]
                            if isinstance(value_range[0], list):
                                real_val = np.random.uniform(value_range[0][0], value_range[0][1])
                                imag_val = np.random.uniform(value_range[1][0], value_range[1][1])
                                params[key] = complex(real_val, imag_val)
                            else:
                                # 只變動實部
                                real_val = np.random.uniform(value_range[0], value_range[1])
                                params[key] = complex(real_val, base_params[key].imag)
                        else:
                            # 其他參數直接隨機採樣
                            params[key] = np.random.uniform(value_range[0], value_range[1])

            return params
        

    def generate_xyt_dataset(self, dataset_2d, n_frames=100, output_dir=None, save_files=True):
        """
        生成 XYT 時間序列數據集 (模擬布朗運動)
        """
        print(f"★ 生成 XYT 時間序列數據集 (Frames={n_frames})...")
        
        # 提取 2D 數據集中的配置
        # 我們從中選擇幾個配置來生成影片
        # dataset_2d 是一個字典，包含 'coords_physical', 'noisy_images' 等
        # 我們主要需要初始粒子座標和粗糙背景
        
        coords_physical_all = dataset_2d['coords_physical'] # [total_samples, N, 3]
        clean_images_all = dataset_2d['clean_images'] # [total_samples, H, W]
        # 背景圖像 (如果有)
        if 'background_images' in dataset_2d:
            background_images_all = dataset_2d['background_images']
        else:
            background_images_all = None
            
        # 選擇一部分樣本來生成影片
        # 假設每個 sample 都是一個獨立的實驗配置
        n_samples = coords_physical_all.shape[0]
        
        xyt_dataset = {
            'movies': [], # List of [T, H, W]
            'trajectories': [], # List of [T, N, 3]
            'metadata': []
        }
        
        import tifffile
        
        for i in tqdm(range(n_samples), desc="生成影片"):
            # 初始狀態
            init_coords = coords_physical_all[i] # [N, 3]
            # 獲取對應的背景 (roughness)
            # 注意：dataset_2d['background_images'] 是縮放後的背景圖像
            # 我們需要原始的 roughness field 來輸入 simulate_with_background
            # 或者我們可以簡單地使用 dataset_2d 中的 metadata 來重建
            # 為了簡單，我們假設 background_images_all[i] 可以被反推，或者我們簡單地使用靜態背景
            
            # 從 metadata 獲取 photon_scale
            meta = dataset_2d['sample_metadata'][i]
            photon_scale = meta.get('photon_scale', 40000)
            
            # 使用 background image / photon_scale 作為 roughness (近似)
            # roughness_shifted = background / scale
            # roughness = roughness_shifted - 1.0
            if background_images_all is not None:
                bg_img = background_images_all[i]
                roughness_field = (bg_img / photon_scale) - 1.0
                roughness_field = torch.tensor(roughness_field, device=self.device, dtype=torch.float32)
            else:
                roughness_field = None
                
            # D = diffusion coefficient
            # Fix: Avoid collision with D_particle (diameter)
            D = self.optical_params.get('diffusion_coeff', 5e-14) # m^2/s, slow diffusion
            dt = 0.01 # seconds
            
            trajectory = self._simulate_brownian_motion(init_coords, n_frames, D, dt)
            # trajectory: [T, N, 3]
            
            movie_frames = []
            
            for t in range(n_frames):
                current_coords = trajectory[t] # [N, 3]
                # 擴展 batch 維度
                current_coords_batch = torch.tensor(current_coords, device=self.device).unsqueeze(0)
                
                # 模擬這一幀
                # 使用相同的 roughness 和 photon_scale
                with torch.no_grad():
                    res = self.optical_model.simulate_with_background(
                        particle_coords=current_coords_batch,
                        roughness_field=roughness_field,
                        add_noise=True,
                        photon_scale=photon_scale,
                        noise_seed=None # Random noise each frame
                    )
                
                movie_frames.append(res['noisy_image'].cpu().numpy()[0])
                
            movie_stack = np.array(movie_frames) # [T, H, W]
            
            xyt_dataset['movies'].append(movie_stack)
            xyt_dataset['trajectories'].append(trajectory)
            xyt_dataset['metadata'].append(meta)
            
            # 保存為 TIF
            if save_files and output_dir:
                filename = f"movie_{i:03d}.tif"
                filepath = os.path.join(output_dir, filename)
                tifffile.imwrite(filepath, movie_stack.astype(np.float32))
                
                # 保存軌跡 GT
                traj_filename = f"movie_{i:03d}_traj.npy"
                np.save(os.path.join(output_dir, traj_filename), trajectory)

        return xyt_dataset

    def _simulate_brownian_motion(self, init_coords, n_frames, D, dt):
        """
        模擬布朗運動
        r(t+dt) = r(t) + sqrt(2*D*dt) * N(0,1)
        """
        n_particles = init_coords.shape[0]
        trajectory = np.zeros((n_frames, n_particles, 3))
        trajectory[0] = init_coords
        
        step_scale = np.sqrt(2 * D * dt)
        
        for t in range(1, n_frames):
            # Z 軸通常 diffusion 較小或是 constrained? 
            # 假設 3D isotropic diffusion
            noise = np.random.normal(0, 1, size=(n_particles, 3))
            step = noise * step_scale
            
            # Update position
            new_pos = trajectory[t-1] + step
            
            # Simple boundary check or constrains?
            # For Z, maybe constrain to be positive
            new_pos[:, 2] = np.abs(new_pos[:, 2]) 
            
            trajectory[t] = new_pos
            
        return trajectory
        
    def generate_integrated_dataset_with_variations(self, particle_coords_list, rough_surfaces, 
                                                   dataset_params, dynamic_ranges=None,
                                                   output_dir=None, output_filename="integrated_dataset", 
                                                   save_files=True):
        """
        生成支援參數變化的數據集

        新增參數:
            dynamic_ranges: 參數變化範圍，例如：
                {
                    'D_particle': [10e-9, 20e-9],
                    'n_particle': [[1.4, 1.5], [2.2, 2.3]],  # [[real範圍], [imag範圍]]
                    'z_focal': [40e-9, 60e-9]
                }
        """
        n_particle_configs = len(particle_coords_list)
        n_rough_surfaces = len(rough_surfaces)
        total_combinations = n_particle_configs * n_rough_surfaces

        print(f"★ 生成整合數據集：")
        print(f"  粒子配置: {n_particle_configs}")
        print(f"  粗糙表面: {n_rough_surfaces}")
        print(f"  總組合數: {total_combinations}")

        # 提取參數
        add_noise = dataset_params.get('add_noise', True)
        photon_scale_range = dataset_params.get('photon_scale_range', (30000, 50000))
        n_samples_per_combo = dataset_params.get('n_samples_per_combo', 1)
        seed = dataset_params.get('seed', 42)

        # ★ 設置隨機種子（移到前面）
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        # 存儲結果
        all_data = {
            'noisy_images': [],
            'clean_images': [],
            'particle_images': [],
            'background_images': [],
            'coords_physical': [],
            'coords_pixel': [],
            'photon_counts': [],
            'particle_z': [],
            'z_focal': [],
            'photon_counts_per_particle': [],
            'contrasts_per_particle': [],
            'metadata_list': []
        }

        sample_idx = 0

        # ★ 只有一個主循環
        for particle_idx in tqdm(range(n_particle_configs), desc="粒子配置進度"):
            particle_coords = particle_coords_list[particle_idx]

            for rough_idx in range(n_rough_surfaces):
                current_roughness = rough_surfaces[rough_idx]

                for sample_in_combo in range(n_samples_per_combo):
                    # ★ 為每個樣本生成新的參數（如果有動態範圍）
                    if dynamic_ranges:
                        # 保存原始參數
                        original_params = self.optical_model.params.copy()

                        # 更新光學模型參數
                        current_params = self.process_dynamic_params(
                            self.optical_model.params, dynamic_ranges
                        )

                        # 臨時更新模型參數
                        for key, value in current_params.items():
                            self.optical_model.params[key] = value
                            # 某些參數可能需要觸發重新計算
                            if key == 'diameter_particle' and hasattr(self.optical_model, '_recompute_mie'):
                                self.optical_model._recompute_mie()

                    # 生成隨機光子縮放
                    photon_scale = random.uniform(photon_scale_range[0], photon_scale_range[1])
                    noise_seed = seed + sample_idx * 1000

                    # 使用當前參數進行仿真
                    result = self.optical_model.simulate_with_background(
                        particle_coords=particle_coords,
                        roughness_field=current_roughness,
                        add_noise=add_noise,
                        photon_scale=photon_scale,
                        noise_seed=noise_seed
                        
                    )

                    # 在pipeline層計算光子數
                    photon_counts = self.estimate_photon_counts(
                        particle_coords, 
                        base_intensity=photon_scale // 10
                    )

                    # 存儲結果
                    all_data['noisy_images'].append(result['noisy_image'].cpu().numpy())
                    all_data['clean_images'].append(result['clean_image'].cpu().numpy())
                    all_data['particle_images'].append(result['particle_image'].cpu().numpy())
                    all_data['background_images'].append(result['background_image'].cpu().numpy())
                    all_data['coords_physical'].append(result['coords_physical'].cpu().numpy()[0]) # Squeeze B=1
                    all_data['coords_pixel'].append(result['coords_pixel'].cpu().numpy()[0]) # Squeeze B=1
                    all_data['photon_counts'].append(photon_counts.cpu().numpy()[0]) # Squeeze B=1
                    all_data['particle_z'].append(result['particle_z'].cpu().numpy()[0]) # Squeeze B=1
                    all_data['z_focal'].append(result['z_focal'])
                    all_data['photon_counts_per_particle'].append(result['photon_counts_per_particle'].cpu().numpy()[0]) # Squeeze B=1
                    all_data['contrasts_per_particle'].append(result['contrasts_per_particle'].cpu().numpy()[0]) # Squeeze B=1

                    # 存儲元數據（包含動態參數）
                    metadata = result['metadata'].copy()
                    metadata.update({
                        'sample_idx': sample_idx,
                        'particle_config_idx': particle_idx,
                        'roughness_idx': rough_idx,
                        'sample_in_combo': sample_in_combo
                    })

                    # ★ 如果有動態參數，記錄實際使用的值
                    if dynamic_ranges:
                        metadata['dynamic_params'] = {
                            key: self.optical_model.params[key] 
                            for key in dynamic_ranges.keys()
                        }
                        # 恢復原始參數
                        self.optical_model.params = original_params

                    all_data['metadata_list'].append(metadata)
                    sample_idx += 1

                # 定期清理GPU記憶體
                if (rough_idx + 1) % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # 轉換為numpy陣列
        for key in ['noisy_images', 'clean_images', 'particle_images', 'background_images',
                   'coords_physical', 'coords_pixel', 'photon_counts', 'particle_z']:
            all_data[key] = np.array(all_data[key])

        # z_focal 是標量，轉換為陣列
        all_data['z_focal'] = np.array(all_data['z_focal'])

        # 組織最終數據集
        dataset = {
            # 圖像數據 [total_samples, B, H, W]
            'noisy_images': all_data['noisy_images'],
            'clean_images': all_data['clean_images'],
            'particle_images': all_data['particle_images'],
            'background_images': all_data['background_images'],

            # 座標和參數數據
            'coords_physical': all_data['coords_physical'],
            'coords_pixel': all_data['coords_pixel'],
            'photon_counts': all_data['photon_counts'],
            'particle_z': all_data['particle_z'],
            'z_focal': all_data['z_focal'],
            'photon_counts_per_particle': all_data['photon_counts_per_particle'],
            'contrasts_per_particle': all_data['contrasts_per_particle'],
            
            # 元數據
            'sample_metadata': all_data['metadata_list'],
            'dataset_metadata': {
                'total_samples': sample_idx,
                'n_particle_configs': n_particle_configs,
                'n_rough_surfaces': n_rough_surfaces,
                'n_samples_per_combo': n_samples_per_combo,
                'image_shape': all_data['noisy_images'].shape[2:],
                'generation_params': dataset_params,
                'optical_params': str(self.optical_params),
                'noise_params': self.noise_params,
                'dynamic_ranges': dynamic_ranges  # 記錄動態範圍
            }
        }

        print(f"\n數據集生成完成！")
        print(f"  總樣本數: {dataset['dataset_metadata']['total_samples']}")
        print(f"  圖像形狀: {dataset['noisy_images'].shape}")
        print(f"  座標形狀: {dataset['coords_physical'].shape}")

        # 保存數據集
        if save_files and output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # 保存主要數據
            npz_path = os.path.join(output_dir, f"{output_filename}.npz")
            save_data = {k: v for k, v in dataset.items() if k != 'sample_metadata'}
            np.savez_compressed(npz_path, **save_data)
            print(f"主要數據已保存: {npz_path}")

            # 保存元數據
            metadata_path = os.path.join(output_dir, f"{output_filename}_metadata.npz")
            np.savez(metadata_path, sample_metadata=dataset['sample_metadata'])
            print(f"元數據已保存: {metadata_path}")

            # 保存示例圖像
            self._save_sample_images(dataset, output_dir, output_filename)

        return dataset
    
    def _save_sample_images(self, dataset, output_dir, filename_prefix):
        """保存示例圖像"""
        sample_dir = os.path.join(output_dir, 'sample_images')
        os.makedirs(sample_dir, exist_ok=True)
        
        # 保存前幾個樣本
        n_samples_to_save = min(5, len(dataset['noisy_images']))
        
        for i in range(n_samples_to_save):
            sample_meta = dataset['sample_metadata'][i]
            particle_idx = sample_meta['particle_config_idx']
            rough_idx = sample_meta['roughness_idx']
            
            # 保存不同類型的圖像
            for img_type in ['noisy_images', 'clean_images', 'particle_images', 'background_images']:
                img = dataset[img_type][i, 0].astype(np.float32)  # 取第一個批次
                
                # 歸一化到0-255
                #img_normalized = img.astype(np.uint8)
                
                img_path = os.path.join(sample_dir, 
                    f"{filename_prefix}_sample{i:02d}_p{particle_idx:02d}_r{rough_idx:02d}_{img_type}.tif")
                tifffile.imwrite(img_path, img)
        
        print(f"示例圖像已保存到: {sample_dir}")

    def visualize_dataset_samples(self, dataset, output_dir, n_samples=3):
        """可視化數據集樣本"""
        vis_dir = os.path.join(output_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        
        n_samples = min(n_samples, len(dataset['noisy_images']))
        
        for i in range(n_samples):
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            sample_meta = dataset['sample_metadata'][i]
            particle_idx = sample_meta['particle_config_idx']
            rough_idx = sample_meta['roughness_idx']
            photon_scale = sample_meta['photon_scale']
            
            # 提取圖像 (取第一個批次)
            noisy_img = dataset['noisy_images'][i, 0]
            clean_img = dataset['clean_images'][i, 0]
            particle_img = dataset['particle_images'][i, 0]
            background_img = dataset['background_images'][i, 0]
            
            # 第一行：原始圖像
            im1 = axes[0, 0].imshow(noisy_img, cmap='viridis')
            axes[0, 0].set_title(f'Noisy Image\n(Particle + Background + Noise)')
            plt.colorbar(im1, ax=axes[0, 0])
            
            im2 = axes[0, 1].imshow(clean_img, cmap='viridis')
            axes[0, 1].set_title(f'Clean Image\n(Particle + Background)')
            plt.colorbar(im2, ax=axes[0, 1])
            
            im3 = axes[0, 2].imshow(particle_img, cmap='plasma')
            axes[0, 2].set_title(f'Particle Only')
            plt.colorbar(im3, ax=axes[0, 2])
            
            # 第二行：分析
            im4 = axes[1, 0].imshow(background_img, cmap='copper')
            axes[1, 0].set_title(f'Background\n(Roughness Surface)')
            plt.colorbar(im4, ax=axes[1, 0])
            
            # 雜訊
            noise_img = noisy_img - clean_img
            im5 = axes[1, 1].imshow(noise_img, cmap='RdBu_r')
            axes[1, 1].set_title(f'Noise\n(Noisy - Clean)')
            plt.colorbar(im5, ax=axes[1, 1])
            
            # 粒子座標疊加
            coords_px = dataset['coords_pixel'][i, 0]  # [N, 3]
            im6 = axes[1, 2].imshow(noisy_img, cmap='viridis')
            if len(coords_px) > 0:
                axes[1, 2].scatter(coords_px[:, 0], coords_px[:, 1], 
                                  c='red', s=20, marker='x', alpha=0.8)
            axes[1, 2].set_title(f'Particle Locations\n({len(coords_px)} particles)')
            plt.colorbar(im6, ax=axes[1, 2])
            
            plt.suptitle(f'Sample {i}: P{particle_idx}-R{rough_idx}, Scale={photon_scale:.0f}', 
                        fontsize=14)
            plt.tight_layout()
            
            # 保存圖像
            plot_path = os.path.join(vis_dir, f'sample_{i:02d}_analysis.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"可視化圖像已保存到: {vis_dir}")

def generate_xyt_dataset(self, base_dataset, n_frames=10, output_dir=None, 
                        output_filename="xyt_dataset", save_files=True):
    """
    從2D數據集生成XYT影片數據集
    
    Args:
        base_dataset: 2D數據集
        n_frames: 每個樣本的幀數
        output_dir: 輸出目錄
        output_filename: 輸出檔名
        save_files: 是否保存檔案
    
    Returns:
        xyt_dataset: XYT格式的數據集
    """
    print(f"\n=== 生成XYT數據集 ===")
    print(f"基礎2D樣本數: {len(base_dataset['clean_images'])}")
    print(f"每個樣本幀數: {n_frames}")
    
    xyt_data = {
        'noisy_movies': [],
        'clean_movies': [],
        'particle_movies': [],
        'background_movies': [],
        'coords_physical': [],
        'coords_pixel': [],
        'photon_counts': [],
        'particle_z': [],
        'z_focal': [],
        'photon_counts_per_particle': [],
        'contrasts_per_particle': [],
        
        'metadata_list': []
    }
    
    # 對每個2D樣本生成影片
    for i in tqdm(range(len(base_dataset['clean_images'])), desc="生成XYT影片"):
        # 提取2D數據 (假設 batch_size = 1)
        clean = base_dataset['clean_images'][i, 0]  # [H, W]
        particle = base_dataset['particle_images'][i, 0]
        background = base_dataset['background_images'][i, 0]
        
        # 座標資訊（每幀相同）
        coords_phys = base_dataset['coords_physical'][i]
        coords_px = base_dataset['coords_pixel'][i]
        photon_cnt = base_dataset['photon_counts'][i]
        p_z = base_dataset['particle_z'][i]
        z_foc = base_dataset['z_focal'][i]
        photon_counts_per_particle = base_dataset['photon_counts_per_particle'][i]
        contrasts_per_particle = base_dataset['contrasts_per_particle'][i]
        #這裡還沒補上contrast
        
        # 原始元數據
        orig_meta = base_dataset['sample_metadata'][i]
        
        # 生成多幀noise
        noisy_frames = []
        for frame_idx in range(n_frames):
            # 使用不同種子生成新的noise
            seed = orig_meta.get('noise_seed', 42) + 10000 * i + frame_idx
            noisy_frame, _ = self.noise_generator.add_poisson_experiment(
                clean, N_scale=1.0, seed=seed
            )
            noisy_frames.append(noisy_frame)
        
        # 組成影片 [T, H, W]
        xyt_data['noisy_movies'].append(np.array(noisy_frames))
        xyt_data['clean_movies'].append(np.repeat(clean[np.newaxis, :, :], n_frames, axis=0))
        xyt_data['particle_movies'].append(np.repeat(particle[np.newaxis, :, :], n_frames, axis=0))
        xyt_data['background_movies'].append(np.repeat(background[np.newaxis, :, :], n_frames, axis=0))
        
        # 座標資訊（保持2D格式的相容性）
        xyt_data['coords_physical'].append(coords_phys)
        xyt_data['coords_pixel'].append(coords_px)
        xyt_data['photon_counts'].append(photon_cnt)
        xyt_data['particle_z'].append(p_z)
        xyt_data['z_focal'].append(z_foc)
        #這裡還沒補上contrast
        xyt_data['photon_counts_per_particle'].append(photon_counts_per_particle)
        xyt_data['contrasts_per_particle'].append(contrasts_per_particle)
        
        # 更新元數據
        xyt_meta = orig_meta.copy()
        xyt_meta['n_frames'] = n_frames
        xyt_meta['xyt_sample_idx'] = i
        xyt_data['metadata_list'].append(xyt_meta)
    
    # 轉為numpy數組
    for key in ['noisy_movies', 'clean_movies', 'particle_movies', 'background_movies']:
        xyt_data[key] = np.array(xyt_data[key])  # [N, T, H, W]
    
    for key in ['coords_physical', 'coords_pixel', 'photon_counts', 'particle_z', 'z_focal','photon_counts_per_particle','contrasts_per_particle']:
        xyt_data[key] = np.array(xyt_data[key])  # 保持原始維度
    
    # 組織最終數據集
    xyt_dataset = {
        # XYT影片數據 [N_samples, T, H, W]
        'noisy_movies': xyt_data['noisy_movies'],
        'clean_movies': xyt_data['clean_movies'],
        'particle_movies': xyt_data['particle_movies'],
        'background_movies': xyt_data['background_movies'],
        
        # 座標和參數（與2D相同格式）
        'coords_physical': xyt_data['coords_physical'],
        'coords_pixel': xyt_data['coords_pixel'],
        'photon_counts': xyt_data['photon_counts'],
        'particle_z': xyt_data['particle_z'],
        'z_focal': xyt_data['z_focal'],
        'photon_counts_per_particle': xyt_data['photon_counts_per_particle'],
        'contrasts_per_particle': xyt_data['contrasts_per_particle'],
        
        # 元數據
        'sample_metadata': xyt_data['metadata_list'],
        'dataset_metadata': {
            'total_samples': len(xyt_data['noisy_movies']),
            'n_frames': n_frames,
            'movie_shape': xyt_data['noisy_movies'].shape[1:],  # [T, H, W]
            'base_dataset_info': base_dataset['dataset_metadata']
        }
    }
    
    print(f"\nXYT數據集生成完成！")
    print(f"  總影片數: {xyt_dataset['dataset_metadata']['total_samples']}")
    print(f"  影片形狀: {xyt_dataset['noisy_movies'].shape}")
    
    # 保存數據集
    if save_files and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存主要數據
        npz_path = os.path.join(output_dir, f"{output_filename}.npz")
        save_data = {k: v for k, v in xyt_dataset.items() if k != 'sample_metadata'}
        np.savez_compressed(npz_path, **save_data)
        print(f"XYT數據已保存: {npz_path}")
        
        # 保存元數據
        metadata_path = os.path.join(output_dir, f"{output_filename}_metadata.npz")
        np.savez(metadata_path, sample_metadata=xyt_dataset['sample_metadata'])
        print(f"XYT元數據已保存: {metadata_path}")
        
        # 保存示例影片（多幀TIFF）
        self._save_xyt_sample_movies(xyt_dataset, output_dir, output_filename)
    
    return xyt_dataset


def _save_xyt_sample_movies(self, dataset, output_dir, filename_prefix):
    """保存XYT示例影片為多幀TIFF"""
    sample_dir = os.path.join(output_dir, 'sample_movies')
    os.makedirs(sample_dir, exist_ok=True)
    
    n_samples_to_save = min(3, len(dataset['noisy_movies']))
    
    for i in range(n_samples_to_save):
        sample_meta = dataset['sample_metadata'][i]
        particle_idx = sample_meta['particle_config_idx']
        rough_idx = sample_meta['roughness_idx']
        
        # 保存不同類型的影片
        for movie_type in ['noisy_movies', 'clean_movies', 'particle_movies', 'background_movies']:
            movie = dataset[movie_type][i].astype(np.float32)  # [T, H, W]
            
            movie_path = os.path.join(sample_dir, 
                f"{filename_prefix}_sample{i:02d}_p{particle_idx:02d}_r{rough_idx:02d}_{movie_type}.tif")
            tifffile.imwrite(movie_path, movie)  # 自動存為多幀TIFF
    
    print(f"XYT示例影片已保存到: {sample_dir}")




def example_with_2d_and_xyt():
    """整合2D和XYT數據集生成的完整範例"""
    
    print("=== 2D + XYT 數據集生成 ===")
    
    # 1. 基礎光學參數配置
    optical_params = {
        'z_focal': 350e-9,
        'n_oil': 1.518,
        't_oil_ideal': 100e-6,
        'n_medium': 1.33,
        'wavelength': 532e-9,
        'n_glass': 1.52,
        'NA': 1.4,
        'nThetas': 64,
        'cam_npixels': 128,
        'cam_pixelsize': 72e-9,
        'geometry': 'iSCAT',
        'z_camera': 0,
        'l': 1,
        'attenuation': 1,
        'scattering_model': 'rayleigh',
        'D_particle': 30e-9,
        'n_particle': 0.5439 + 2.2309*1j,
        'zernike_coeffs': 0,
        'apply_polar_mod': True,
        'apply_coherence': False,
        'polarization': torch.tensor([1.0, 1j], dtype=torch.complex128) / torch.sqrt(torch.tensor(2.0)),
        'background_mode': 'roughness',
        'gain': 2.17,
        'offset': 100,
        'read_noise': 2.0,
        
        'qwp_angle': math.pi/4*4,
        'spatial_coherence_length': 5e-7,
        'wavelength_bandwidth_nm': 10,
        
        'flip_phasemask_lr': True,
        'flip_es_ud': True
    }
    
    # 2. 創建仿真管道
    noise_params = {'gain': 2.17, 'offset': 100, 'read_noise': 2.0}
    pipeline = EnhancedSimulationPipeline(optical_params, noise_params)
    
    # 為pipeline添加XYT生成方法
    pipeline.generate_xyt_dataset = generate_xyt_dataset.__get__(pipeline, EnhancedSimulationPipeline)
    pipeline._save_xyt_sample_movies = _save_xyt_sample_movies.__get__(pipeline, EnhancedSimulationPipeline)
    
    # 3. 生成粗糙表面
    print("\n步驟 1: 生成粗糙表面")
    rough_surfaces, _ = pipeline.generate_rough_surfaces(
        n_surfaces=10,
        surface_params={
            'N': 128,
            'pixel_size': 72,
            'sigma': 0.002,
            'xi': 100
        },
        save_files=False
    )
    
    # 4. 動態參數範圍（可選）
    dynamic_ranges = {
        'D_particle': [100e-9, 100e-9],
        'n_particle': [[1.44, 1.46], [0.1, 0.1]],
        'z_focal': [300e-9, 350e-9]
    }
    
    # 5. 生成粒子座標
    print("\n步驟 2: 生成粒子座標")
    particle_coords_list = pipeline.generate_particle_coordinates(
        n_configs=10,
        generation_params={
            'type': 'random',
            'n_particles': 10,
            'field_of_view': optical_params['cam_npixels'] * optical_params['cam_pixelsize'] / 2 * 0.8,
            'z_range': [70e-9, 70e-9]
        }
    )
    
    # 6. 數據集參數
    dataset_params = {
        'add_noise': True,
        'photon_scale_range': (40000, 40000),
        'n_samples_per_combo': 1,  # 2D數據集每組合1個樣本
        'seed': 42
    }
    
    # 7. 生成2D數據集
    print("\n步驟 3: 生成2D數據集")
    dataset_2d = pipeline.generate_integrated_dataset_with_variations(
        particle_coords_list=particle_coords_list,
        rough_surfaces=rough_surfaces,
        dataset_params=dataset_params,
        dynamic_ranges=dynamic_ranges,
        output_dir='output/2d_dataset',
        output_filename='dataset_2d',
        save_files=True
    )
    
    # 8. 從2D生成XYT數據集
    print("\n步驟 4: 從2D數據集生成XYT影片數據集")
    dataset_xyt = pipeline.generate_xyt_dataset(
        base_dataset=dataset_2d,
        n_frames=3,  # 每個樣本20幀
        output_dir='output/xyt_dataset',
        output_filename='dataset_xyt',
        save_files=True
    )
    
    # 9. 總結
    print("\n=== 數據集生成完成 ===")
    print(f"\n2D數據集:")
    print(f"  總樣本數: {dataset_2d['dataset_metadata']['total_samples']}")
    print(f"  圖像形狀: {dataset_2d['noisy_images'].shape}")
    
    print(f"\nXYT數據集:")
    print(f"  總影片數: {dataset_xyt['dataset_metadata']['total_samples']}")
    print(f"  影片形狀: {dataset_xyt['noisy_movies'].shape}")
    print(f"  每影片幀數: {dataset_xyt['dataset_metadata']['n_frames']}")
    
    return dataset_2d, dataset_xyt, pipeline


if __name__ == "__main__":
    # 運行整合版本（同時生成2D和XYT）
    dataset_2d, dataset_xyt, pipeline = example_with_2d_and_xyt()
    
    print("\n完成！檢查以下目錄：")
    print("  - output/2d_dataset/   (2D數據集)")
    print("  - output/xyt_dataset/  (XYT影片數據集)")


# In[ ]:






if __name__ == '__main__':
    # In[4]:
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import os
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'STSong', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['image.origin'] = 'lower'   # 或 'upper'
    
    # 載入數據
    data_path = 'output/2d_dataset/dataset_2d.npz'
    metadata_path = 'output/2d_dataset/dataset_2d_metadata.npz'
    
    # 載入主要數據
    with np.load(data_path) as data:
        noisy_images = data['noisy_images']
        clean_images = data['clean_images']
        particle_images = data['particle_images']
        background_images = data['background_images']
        coords_pixel = data['coords_pixel']
        coords_physical = data['coords_physical']
        photon_counts = data['photon_counts']
        particle_z = data['particle_z']
        z_focal = data['z_focal']
        contrasts = data['contrasts_per_particle']
    
            
    # 載入元數據
    with np.load(metadata_path, allow_pickle=True) as meta:
        sample_metadata = meta['sample_metadata']
    
    # 打印數據集資訊
    print("=== 數據集資訊 ===")
    print(f"總樣本數: {len(noisy_images)}")
    print(f"圖像形狀: {noisy_images.shape}")
    print(f"座標形狀: {coords_pixel.shape}")
    print(f"光子計數形狀: {photon_counts.shape}")
    
    # 檢查前3筆數據
    n_samples_to_check = min(3, len(noisy_images))
    
    for sample_idx in range(n_samples_to_check):
        print(f"\n{'='*60}")
        print(f"樣本 #{sample_idx}")
        print(f"{'='*60}")
        
        # 獲取元數據
        meta = sample_metadata[sample_idx]
        particle_idx = meta['particle_config_idx']
        rough_idx = meta['roughness_idx']
        photon_scale = meta['photon_scale']
        
        print(f"\n元數據:")
        print(f"  粒子配置索引: {particle_idx}")
        print(f"  粗糙表面索引: {rough_idx}")
        print(f"  光子縮放: {photon_scale:.0f}")
        
        # 提取當前樣本的數據 (假設 batch size = 1，取第一個 batch)
        noisy = noisy_images[sample_idx, 0]
        clean = clean_images[sample_idx, 0]
        particle = particle_images[sample_idx, 0]
        background = background_images[sample_idx, 0]
        
        # 計算統計資訊
        print(f"\n各通道統計:")
        print(f"  Noisy - 最小值: {noisy.min():.2f}, 最大值: {noisy.max():.2f}, 平均值: {noisy.mean():.2f}")
        print(f"  Clean - 最小值: {clean.min():.2f}, 最大值: {clean.max():.2f}, 平均值: {clean.mean():.2f}")
        print(f"  Particle - 最小值: {particle.min():.2f}, 最大值: {particle.max():.2f}, 平均值: {particle.mean():.2f}")
        print(f"  Background - 最小值: {background.min():.2f}, 最大值: {background.max():.2f}, 平均值: {background.mean():.2f}")
        
        # 粒子座標資訊
        coords = coords_pixel[sample_idx, 0]  # [N, 3]
        #photons = photon_counts[sample_idx, 0]  # [N]
        # 使用對比度和光子數
        if contrasts !=[]:
            photons = contrasts*photon_scale  # [N]
            
            print(f"\n粒子資訊:")
            print(f"  粒子數量: {len(coords)}")
            print(f"  對比度範圍: [{contrasts.min():.4f}, {contrasts.max():.4f}]")
            print(f"  光子數範圍: [{photons.min():.0f}, {photons.max():.0f}]")
        else:
            # 舊版本兼容
            photons = photon_counts[sample_idx, 0]
            print(f"  光子數範圍: [{photons.min():.0f}, {photons.max():.0f}]")
        
        
        z_coords = particle_z[sample_idx, 0]  # [N]
        
        print(f"\n粒子資訊:")
        print(f"  粒子數量: {len(coords)}")
        print(f"  X座標範圍: [{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}]")
        print(f"  Y座標範圍: [{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}]")
        print(f"  Z座標範圍: [{z_coords.min():.1f}, {z_coords.max():.1f}] nm")
        print(f"  光子數範圍: [{photons.min():.0f}, {photons.max():.0f}]")
        
        # 創建視覺化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 第一行：原始圖像
        # Noisy image
        im1 = axes[0, 0].imshow(noisy, cmap='viridis')
        axes[0, 0].set_title(f'Noisy Image\nMin: {noisy.min():.1f}, Max: {noisy.max():.1f}')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # Clean image
        im2 = axes[0, 1].imshow(clean, cmap='viridis')
        axes[0, 1].set_title(f'Clean Image\nMin: {clean.min():.1f}, Max: {clean.max():.1f}')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Particle image
        im3 = axes[0, 2].imshow(particle, cmap='plasma')
        axes[0, 2].set_title(f'Particle Only\nMin: {particle.min():.1f}, Max: {particle.max():.1f}')
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # 第二行：分析
        # Background
        im4 = axes[1, 0].imshow(background, cmap='copper')
        axes[1, 0].set_title(f'Background\nMin: {background.min():.1f}, Max: {background.max():.1f}')
        plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Noise (difference)
        noise = noisy - clean
        im5 = axes[1, 1].imshow(noise, cmap='RdBu_r', 
                                vmin=-np.abs(noise).max(), vmax=np.abs(noise).max())
        axes[1, 1].set_title(f'Noise (Noisy - Clean)\nMin: {noise.min():.1f}, Max: {noise.max():.1f}')
        plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        H = noisy.shape[0]
        # 粒子位置疊加
        im6 = axes[1, 2].imshow(noisy, cmap='gray', origin='lower')
        # 繪製粒子位置
        for i, (x, y, z) in enumerate(coords):
            # 用紅色 X 標記粒子位置
            y_flip = H - y  # 将 y 从“自底向上”坐标 → “自顶向下”
            axes[1, 2].plot(x, y_flip, 'rx', markersize=10, markeredgewidth=2)
            # 添加粒子編號
            axes[1, 2].text(x+2, y_flip+2, f'{i}', color='yellow', fontsize=8, 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
        
        axes[1, 2].set_title(f'Particle Locations ({len(coords)} particles)')
        axes[1, 2].set_xlim(0, noisy.shape[1])
        axes[1, 2].set_ylim(0, noisy.shape[0])  # 翻轉Y軸以匹配圖像座標
        plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        # 設置整體標題
        plt.suptitle(f'Sample #{sample_idx}: Particle Config {particle_idx} × Rough Surface {rough_idx}\n' + 
                     f'Photon Scale: {photon_scale:.0f}, Z focal: {z_focal[sample_idx]:.2e} m', 
                     fontsize=14)
        
        plt.tight_layout()
        plt.show()
        
        # 額外的詳細粒子資訊圖
        if len(coords) > 0:
            fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    
            # 確保 photons 是當前樣本的數據
            # 如果 photon_counts 的形狀是 [total_samples, B, N]
            if photons.ndim == 3:
                current_photons = photons[sample_idx, 0]  # [N]
            # 如果是 [total_samples, N] （假設 B=1）
            elif photon_counts.ndim == 2:
                current_photons = photons[sample_idx]  # [N]
            else:
                # 兼容舊格式
                current_photons = photons
    
            # 檢查維度是否匹配
            if len(current_photons) != len(coords):
                print(f"警告：光子數維度 ({len(current_photons)}) 與座標數量 ({len(coords)}) 不匹配")
                # 嘗試修正或使用默認值
                if len(current_photons) > len(coords):
                    current_photons = current_photons[:len(coords)]
                else:
                    # 使用固定值作為後備
                    current_photons = np.ones(len(coords)) * 4000
    
            # 粒子位置散點圖（帶大小和顏色）
            # 確保 s 參數的大小合理
            sizes = current_photons / np.max(current_photons) * 300 + 50  # 歸一化到 50-350 的範圍
    
            scatter = axes2[0].scatter(coords[:, 0], coords[:, 1], 
                                      s=sizes,  # 大小代表光子數（歸一化）
                                      c=z_coords,     # 顏色代表Z座標
                                      cmap='coolwarm', 
                                      alpha=0.6,
                                      edgecolors='black',
                                      linewidths=1)
            axes2[0].set_xlabel('X (pixels)')
            axes2[0].set_ylabel('Y (pixels)')
            axes2[0].set_title(f'粒子分布\n(大小∝光子數, 顏色=Z座標)')
            axes2[0].set_xlim(0, noisy.shape[1])
            axes2[0].set_ylim(noisy.shape[0], 0)
            axes2[0].grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=axes2[0])
            cbar.set_label('Z coordinate (nm)')
    
            # 添加圖例說明大小比例
            # 創建假的散點作為圖例
            for photon_val in [current_photons.min(), current_photons.mean(), current_photons.max()]:
                axes2[0].scatter([], [], s=photon_val/np.max(current_photons)*300+50, 
                                c='gray', alpha=0.6, edgecolors='black',
                                label=f'{photon_val:.0f} photons')
            axes2[0].legend(scatterpoints=1, frameon=True, labelspacing=1, 
                           title='Photon Count', loc='upper right', fontsize=8)
    
            # 粒子統計直方圖
            axes2[1].hist(current_photons, bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes2[1].set_xlabel('光子數')
            axes2[1].set_ylabel('粒子數量')
            axes2[1].set_title(f'光子數分布\n平均: {current_photons.mean():.1f}, 標準差: {current_photons.std():.1f}')
            axes2[1].grid(True, alpha=0.3)
    
            # 添加統計信息
            axes2[1].axvline(current_photons.mean(), color='red', linestyle='--', 
                            label=f'平均值: {current_photons.mean():.1f}')
            axes2[1].axvline(np.median(current_photons), color='green', linestyle='--', 
                            label=f'中位數: {np.median(current_photons):.1f}')
            axes2[1].legend()
    
            plt.suptitle(f'Sample #{sample_idx} - 詳細粒子資訊', fontsize=14)
            plt.tight_layout()
            plt.show()
    
            # 打印詳細統計
            print(f"\n詳細光子數統計:")
            print(f"  最小值: {current_photons.min():.1f}")
            print(f"  最大值: {current_photons.max():.1f}")
            print(f"  平均值: {current_photons.mean():.1f}")
            print(f"  中位數: {np.median(current_photons):.1f}")
            print(f"  標準差: {current_photons.std():.1f}")
    
        print("\n" + "-"*60)
    
    
    # In[7]:
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # 設置全局參數
    plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'STSong', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['image.origin'] = 'lower'  # 圖像原點在左下角
    
    # 資料路徑
    data_path = 'output/2d_dataset/dataset_2d.npz'
    metadata_path = 'output/2d_dataset/dataset_2d_metadata.npz'
    
    # 載入主要數據
    with np.load(data_path) as data:
        noisy_images = data['noisy_images']
        clean_images = data['clean_images']
        particle_images = data['particle_images']
        background_images = data['background_images']
        coords_pixel = data['coords_pixel']
        coords_physical = data['coords_physical']
        photon_counts = data['photon_counts']
        particle_z = data['particle_z']
        z_focal = data['z_focal']
        contrasts = data['contrasts_per_particle']
    
    # 載入元數據
    with np.load(metadata_path, allow_pickle=True) as meta:
        sample_metadata = meta['sample_metadata']
    
    # 打印數據集資訊
    print("=== 數據集資訊 ===")
    print(f"總樣本數: {len(noisy_images)}")
    print(f"圖像形狀: {noisy_images.shape}")
    print(f"像素座標形狀: {coords_pixel.shape}")
    print(f"物理座標形狀: {coords_physical.shape}")
    print(f"光子計數形狀: {photon_counts.shape}")
    print(f"粒子Z形狀: {particle_z.shape}")
    print(f"Z焦點形狀: {z_focal.shape}")
    print(f"對比度形狀: {contrasts.shape if contrasts is not None else 'None'}")
    
    # 檢查前3筆數據
    n_samples_to_check = min(3, len(noisy_images))
    output_dir = 'output/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    for sample_idx in range(n_samples_to_check):
        print(f"\n{'='*60}")
        print(f"樣本 #{sample_idx}")
        print(f"{'='*60}")
    
        # 獲取元數據
        meta = sample_metadata[sample_idx]
        particle_idx = meta['particle_config_idx']
        rough_idx = meta['roughness_idx']
        photon_scale = meta['photon_scale']
        
        print(f"\n元數據:")
        print(f"  粒子配置索引: {particle_idx}")
        print(f"  粗糙表面索引: {rough_idx}")
        print(f"  光子縮放: {photon_scale:.0f}")
    
        # 提取當前樣本的數據 (假設 batch size = 1，取第一個 batch)
        noisy = noisy_images[sample_idx, 0]
        clean = clean_images[sample_idx, 0]
        particle = particle_images[sample_idx, 0]
        background = background_images[sample_idx, 0]
        coords_pix = coords_pixel[sample_idx, 0]  # [N, 3]
        coords_phys = coords_physical[sample_idx, 0]  # [N, 3]
        z_coords = particle_z[sample_idx, 0]  # [N]
        z_foc = z_focal[sample_idx]  # 標量
        
        # 處理光子數與對比度
        if contrasts is not None and len(contrasts[sample_idx, 0]) > 0:
            cont = contrasts[sample_idx, 0]  # [N]
            photons = cont * photon_scale  # [N]
        else:
            photons = photon_counts[sample_idx, 0]  # [N]
            print("警告：未找到對比度資料，使用原始光子數")
    
        # 檢查座標範圍
        H, W = noisy.shape
        print(f"\n座標資訊:")
        print(f"  像素座標 X 範圍: [{coords_pix[:, 0].min():.1f}, {coords_pix[:, 0].max():.1f}]")
        print(f"  像素座標 Y 範圍: [{coords_pix[:, 1].min():.1f}, {coords_pix[:, 1].max():.1f}]")
        print(f"  圖像尺寸: {W}x{H}")
        print(f"  物理座標 X 範圍: [{coords_phys[:, 0].min():.1f}, {coords_phys[:, 0].max():.1f}]")
        print(f"  物理座標 Y 範圍: [{coords_phys[:, 1].min():.1f}, {coords_phys[:, 1].max():.1f}]")
        print(f"  Z 座標範圍: [{z_coords.min():.1f}, {z_coords.max():.1f}] nm")
        print(f"  Z 焦點: {z_foc:.2e} m")
        print(f"  光子數範圍: [{photons.min():.0f}, {photons.max():.0f}]")
    
        # 第一組圖：圖像數據與粒子位置
        fig1, axes1 = plt.subplots(2, 2, figsize=(12, 12))
        
        # Noisy Image
        im1 = axes1[0, 0].imshow(noisy, cmap='viridis', origin='lower')
        axes1[0, 0].set_title(f'Noisy Image\nMin: {noisy.min():.1f}, Max: {noisy.max():.1f}')
        plt.colorbar(im1, ax=axes1[0, 0], fraction=0.046, pad=0.04)
        
        # Clean Image
        im2 = axes1[0, 1].imshow(clean, cmap='viridis', origin='lower')
        axes1[0, 1].set_title(f'Clean Image\nMin: {clean.min():.1f}, Max: {clean.max():.1f}')
        plt.colorbar(im2, ax=axes1[0, 1], fraction=0.046, pad=0.04)
        
        # Particle Image
        im3 = axes1[1, 0].imshow(particle, cmap='plasma', origin='lower')
        axes1[1, 0].set_title(f'Particle Only\nMin: {particle.min():.1f}, Max: {particle.max():.1f}')
        plt.colorbar(im3, ax=axes1[1, 0], fraction=0.046, pad=0.04)
        
        # Background Image
        im4 = axes1[1, 1].imshow(background, cmap='copper', origin='lower')
        axes1[1, 1].set_title(f'Background\nMin: {background.min():.1f}, Max: {background.max():.1f}')
        plt.colorbar(im4, ax=axes1[1, 1], fraction=0.046, pad=0.04)
        
        # 疊加粒子位置（檢查翻轉）
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        im5 = ax2.imshow(noisy, cmap='gray', origin='lower')
        for i, (x, y, _) in enumerate(coords_pix):
            # 根據 Y 座標範圍判斷是否翻轉
            y_plot = H - y if coords_pix[:, 1].max() > H / 2 else y
            ax2.plot(x, y_plot, 'rx', markersize=10, markeredgewidth=2)
            ax2.text(x+2, y_plot+2, f'{i}', color='yellow', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
        ax2.set_title(f'Particle Locations ({len(coords_pix)} particles)\nY Flip: {coords_pix[:, 1].max() > H / 2}')
        ax2.set_xlim(0, W)
        ax2.set_ylim(0, H)
        plt.colorbar(im5, ax=ax2, fraction=0.046, pad=0.04)
        
        # 設置主圖標題
        plt.suptitle(f'Sample #{sample_idx}: Particle Config {particle_idx} × Rough Surface {rough_idx}\n'
                     f'Photon Scale: {photon_scale:.0f}, Z Focal: {z_foc:.2e} m', fontsize=14)
        plt.tight_layout()
        fig1.savefig(os.path.join(output_dir, f'sample_{sample_idx}_images.png'), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(output_dir, f'sample_{sample_idx}_particles.png'), dpi=300, bbox_inches='tight')
        plt.close(fig1)
        plt.close(fig2)
    
        # 第二組圖：詳細數據分析
        if len(coords_pix) > 0:
            fig3, axes3 = plt.subplots(2, 2, figsize=(12, 12))
            
            # 像素座標散點圖
            sizes = photons / np.max(photons) * 300 + 50
            scatter1 = axes3[0, 0].scatter(coords_pix[:, 0], coords_pix[:, 1], 
                                           s=sizes, c=z_coords, cmap='coolwarm', 
                                           alpha=0.6, edgecolors='black', linewidths=1)
            axes3[0, 0].set_xlabel('X (pixels)')
            axes3[0, 0].set_ylabel('Y (pixels)')
            axes3[0, 0].set_title('Pixel Coordinates\n(Size ∝ Photons, Color = Z)')
            axes3[0, 0].set_xlim(0, W)
            axes3[0, 0].set_ylim(0, H)
            plt.colorbar(scatter1, ax=axes3[0, 0], label='Z (nm)')
            
            # 物理座標散點圖
            scatter2 = axes3[0, 1].scatter(coords_phys[:, 0], coords_phys[:, 1], 
                                           s=sizes, c=z_coords, cmap='coolwarm', 
                                           alpha=0.6, edgecolors='black', linewidths=1)
            axes3[0, 1].set_xlabel('X (physical)')
            axes3[0, 1].set_ylabel('Y (physical)')
            axes3[0, 1].set_title('Physical Coordinates\n(Size ∝ Photons, Color = Z)')
            plt.colorbar(scatter2, ax=axes3[0, 1], label='Z (nm)')
            
            # 光子數直方圖
            axes3[1, 0].hist(photons, bins=20, color='blue', alpha=0.7, edgecolor='black')
            axes3[1, 0].set_xlabel('Photon Number')
            axes3[1, 0].set_ylabel('Count')
            axes3[1, 0].set_title(f'Photon Counts\nMean: {np.mean(photons):.1f}, Std: {np.std(photons):.1f}')
            axes3[1, 0].axvline(np.mean(photons), color='red', linestyle='--', label=f'Mean: {np.mean(photons):.1f}')
            axes3[1, 0].legend()
            
            # Z 座標直方圖
            axes3[1, 1].hist(z_coords, bins=20, color='green', alpha=0.7, edgecolor='black')
            axes3[1, 1].set_xlabel('Z Coordinate (nm)')
            axes3[1, 1].set_ylabel('Count')
            axes3[1, 1].set_title(f'Z Coordinates\nMean: {np.mean(z_coords):.1f}, Std: {np.std(z_coords):.1f}')
            axes3[1, 1].axvline(np.mean(z_coords), color='red', linestyle='--', label=f'Mean: {np.mean(z_coords):.1f}')
            axes3[1, 1].legend()
            
            plt.suptitle(f'Sample #{sample_idx}: Detailed Data Analysis', fontsize=14)
            plt.tight_layout()
            fig3.savefig(os.path.join(output_dir, f'sample_{sample_idx}_detailed.png'), dpi=300, bbox_inches='tight')
            plt.close(fig3)
    
        print("\n" + "-"*60)
    
    
    # In[2]:
    
    
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 創建測試數據
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模擬參數
    photon_scale = 40000
    image_size = 128
    
    # 1. 創建測試粗糙表面（小的變化）
    roughness = torch.randn(image_size, image_size, device=device) * 0.001  # 很小的變化
    
    # 2. 創建測試粒子強度（假設有幾個亮點）
    particle_intensity = torch.zeros(image_size, image_size, device=device)
    # 添加幾個粒子信號
    particle_intensity[30, 40] = 0.01  # 小信號
    particle_intensity[60, 70] = 0.02
    particle_intensity[80, 50] = 0.015
    
    # 3. 應用正確的公式
    print("=== 測試正確的仿真公式 ===")
    
    # 偏移到1
    roughness_shifted = roughness + 1.0
    particle_shifted = particle_intensity + 1.0
    
    print(f"粗糙度偏移後 - 最小值: {roughness_shifted.min():.4f}, 最大值: {roughness_shifted.max():.4f}, 平均值: {roughness_shifted.mean():.4f}")
    print(f"粒子偏移後 - 最小值: {particle_shifted.min():.4f}, 最大值: {particle_shifted.max():.4f}, 平均值: {particle_shifted.mean():.4f}")
    
    # 計算clean圖像
    clean_normalized = (roughness_shifted + particle_shifted) / 2.0
    clean_total = clean_normalized * photon_scale
    
    print(f"\nClean圖像 - 最小值: {clean_total.min():.1f}, 最大值: {clean_total.max():.1f}, 平均值: {clean_total.mean():.1f}")
    print(f"預期範圍: 約 {photon_scale} 附近")
    
    # 添加泊松雜訊
    clean_numpy = clean_total.cpu().numpy()
    noisy = np.random.poisson(clean_numpy)
    
    print(f"\nNoisy圖像 - 最小值: {noisy.min():.1f}, 最大值: {noisy.max():.1f}, 平均值: {noisy.mean():.1f}")
    
    # 視覺化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始數據
    im1 = axes[0, 0].imshow(roughness.cpu(), cmap='RdBu_r')
    axes[0, 0].set_title(f'原始粗糙度\n範圍: [{roughness.min():.4f}, {roughness.max():.4f}]')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(particle_intensity.cpu(), cmap='plasma')
    axes[0, 1].set_title(f'原始粒子強度\n範圍: [{particle_intensity.min():.4f}, {particle_intensity.max():.4f}]')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 偏移後
    im3 = axes[0, 2].imshow(roughness_shifted.cpu(), cmap='viridis')
    axes[0, 2].set_title(f'粗糙度+1\n範圍: [{roughness_shifted.min():.4f}, {roughness_shifted.max():.4f}]')
    plt.colorbar(im3, ax=axes[0, 2])
    
    im4 = axes[1, 0].imshow(particle_shifted.cpu(), cmap='viridis')
    axes[1, 0].set_title(f'粒子+1\n範圍: [{particle_shifted.min():.4f}, {particle_shifted.max():.4f}]')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # 最終圖像
    im5 = axes[1, 1].imshow(clean_total.cpu(), cmap='gray')
    axes[1, 1].set_title(f'Clean (縮放後)\n範圍: [{clean_total.min():.0f}, {clean_total.max():.0f}]')
    plt.colorbar(im5, ax=axes[1, 1])
    
    im6 = axes[1, 2].imshow(noisy, cmap='gray')
    axes[1, 2].set_title(f'Noisy (泊松後)\n範圍: [{noisy.min():.0f}, {noisy.max():.0f}]')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.suptitle(f'仿真流程測試 (photon_scale = {photon_scale})', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 驗證統計特性
    print("\n=== 統計驗證 ===")
    print(f"Clean平均值 / photon_scale = {clean_total.mean().item() / photon_scale:.4f} (應該接近1)")
    print(f"Noisy平均值 / Clean平均值 = {noisy.mean() / clean_total.mean().item():.4f} (應該接近1)")
    print(f"Noisy變異數 / Noisy平均值 = {noisy.var() / noisy.mean():.4f} (泊松分布應該接近1)")
    
    # 檢查縮放後的粒子和背景
    particle_scaled = particle_shifted * photon_scale
    background_scaled = roughness_shifted * photon_scale
    
    print(f"\n縮放後的粒子圖像 - 範圍: [{particle_scaled.min():.0f}, {particle_scaled.max():.0f}]")
    print(f"縮放後的背景圖像 - 範圍: [{background_scaled.min():.0f}, {background_scaled.max():.0f}]")
    
    
    # In[ ]:
    
    
    
    
