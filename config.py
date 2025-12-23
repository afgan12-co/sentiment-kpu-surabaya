"""
Configuration file untuk kategori keyword detection
"""

# Keyword untuk deteksi kategori
CATEGORY_KEYWORDS = {
    'kinerja': [
        'pelayanan', 'cepat', 'efisien', 'kinerja', 'petugas', 'administrasi',
        'proses', 'layanan', 'response', 'tanggap', 'profesional', 'ramah',
        'staff', 'pegawai', 'tim', 'antrian', 'waktu', 'lama', 'lambat'
    ],
    'netralitas': [
        'netral', 'independen', 'adil', 'objektif', 'transparan', 'jujur',
        'bebas', 'tidak memihak', 'fair', 'imparsial', 'seimbang', 'terbuka',
        'akuntabel', 'integritas', 'kredibel', 'kepercayaan'
    ],
    'kebijakan': [
        'pemilu', 'regulasi', 'kebijakan', 'peraturan', 'undang', 'aturan',
        'hukum', 'sistem', 'prosedur', 'persyaratan', 'ketentuan', 'mekanisme',
        'program', 'kampanye', 'pilkada', 'voting', 'suara', 'tps', 'dpt'
    ]
}

# Model preference untuk prediction (default: 'nb' untuk Naive Bayes)
DEFAULT_MODEL = 'nb'  # 'nb' atau 'svm'
