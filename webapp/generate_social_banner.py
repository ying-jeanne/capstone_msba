from PIL import Image, ImageDraw, ImageFont
import os

def create_banner():
    # Configuration
    WIDTH = 1200
    HEIGHT = 630
    
    # EXACT Brand Dark Slate (Sampled from logo: 30, 41, 59)
    # This ensures the banner background matches the logo's dark lines perfectly.
    BG_COLOR_RGB = (30, 41, 59) 
    
    TEXT_COLOR_MAIN = "#ffffff"  # White
    TEXT_COLOR_SUB = "#f59e0b"   # Amber (Brand Accent)
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOGO_PATH = os.path.join(BASE_DIR, "static/images/favicon.png")
    OUTPUT_PATH = os.path.join(BASE_DIR, "static/images/social_banner.png")
    
    # Create background with EXACT RGB
    img = Image.new('RGB', (WIDTH, HEIGHT), color=BG_COLOR_RGB)
    draw = ImageDraw.Draw(img)
    
    # Load and Resize Logo
    try:
        logo = Image.open(LOGO_PATH).convert("RGBA")
        
        # Determine Logo Size
        logo_size = int(HEIGHT * 0.55)
        logo = logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
        
        # Create a new image for the processed logo
        data = logo.getdata()
        new_data = []
        
        # De-matting Logic:
        # The logo has anti-aliased edges against white.
        # We want to remove the white but keep the alpha.
        # A pixel P = C * alpha + White * (1-alpha)
        # If we assume C is our Dark Slate color (30, 41, 59), we can solve for alpha.
        # However, we also have Amber pixels.
        
        target_dark = (30, 41, 59)
        
        for item in data:
            r, g, b, a = item
            
            # Helper values
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            saturation = max_val - min_val
            lightness = max_val # Simple brightness proxy
            
            # LOGIC:
            # 1. Dark Logo pixels: Low Saturation, Low Lightness (Dark Slate)
            # 2. Amber Logo pixels: High Saturation (Amber)
            # 3. Halo/Background pixels: Low Saturation, High Lightness (White/Grey)
            
            # Thresholds:
            # Dark Slate is approx (30,41,59). Max ~60. Sat ~30.
            # Amber is (245,158,11). Sat ~234.
            # Halo pixels are usually > 100 brightness and low saturation.
            
            if saturation < 40:
                # Low saturation -> It's either Dark Logo or White/Grey Halo
                
                if lightness < 100:
                    # It's DARK (The logo text/icon lines). 
                    # We apply the de-matting math to ensure smooth edges against the dark banner.
                    # Dark Slate Green channel is ~41.
                    if g <= 41:
                        new_alpha = 255
                    elif g >= 255:
                        new_alpha = 0
                    else:
                        new_alpha = int(255 * (1.0 - (g - 41) / (214.0)))
                    
                    new_alpha = max(0, min(255, new_alpha))
                    new_data.append((target_dark[0], target_dark[1], target_dark[2], new_alpha))
                    
                else:
                    # It's LIGHT (Grey/White Halo) -> Remove it completely
                    new_data.append((255, 255, 255, 0))
            else:
                # High saturation -> It's the Amber part or colorful noise.
                # Keep it original.
                new_data.append(item)
                
        logo.putdata(new_data)
        
        # Position Logo (Left side with padding)
        logo_x = 100
        logo_y = (HEIGHT - logo_size) // 2
        
        logo.putdata(new_data)
        
        # Position Logo (Left side with padding)
        logo_x = 100
        logo_y = (HEIGHT - logo_size) // 2
        
        # SAVE THE PROCESSED LOGO (Useful for Portfolio/Personal Website)
        # User requested ROUND BORDER (Circular Mask)
        
        # 1. Create a copy for the portfolio
        portfolio_logo = logo.copy()
        
        # 2. Create a circular mask
        # We start with a blank (black) mask
        mask = Image.new("L", portfolio_logo.size, 0)
        draw_mask = ImageDraw.Draw(mask)
        # Draw white circle
        draw_mask.ellipse((0, 0) + portfolio_logo.size, fill=255)
        
        # 3. Combine with existing alpha (intersection)
        # New Alpha = Existing Alpha * Circle Mask / 255
        # We need ImageChops for this math
        from PIL import ImageChops
        existing_alpha = portfolio_logo.split()[3]
        final_alpha = ImageChops.multiply(existing_alpha, mask)
        
        # 4. Apply new alpha
        portfolio_logo.putalpha(final_alpha)
        
        clean_logo_path = os.path.join(BASE_DIR, "static/images/project_logo.png")
        portfolio_logo.save(clean_logo_path)
        print(f"✅ Cleaned ROUND logo saved to: {clean_logo_path}")
        
        # Paste Logo using itself as mask
        img.paste(logo, (logo_x, logo_y), logo)
            
    except Exception as e:
        print(f"Error loading logo: {e}")
        return

    # Add Text
    # For a robust script without external font dependencies, we'll try to load a system font
    # If not found, we fall back to default (which is small, but functional)
    
    try:
        # Try common fonts on macOS/Linux
        font_path_main = "/System/Library/Fonts/Helvetica.ttc"
        if not os.path.exists(font_path_main):
            font_path_main = "/System/Library/Fonts/Supplemental/Arial.ttf"
            
        # Font Sizes
        size_main = 70
        size_sub = 40
        
        font_main = ImageFont.truetype(font_path_main, size_main, index=0)
        font_sub = ImageFont.truetype(font_path_main, size_sub, index=0)
        
    except Exception:
        print("Warning: Custom fonts not found, using default.")
        font_main = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    # Text Content
    title_text = "Verifiable\nBitcoin Insights"
    subtitle_text = "Transparent AI Predictions\nVerified on Blockchain"
    
    # Position Text (Right of logo)
    text_x = logo_x + logo_size + 60
    text_y_start = (HEIGHT // 2) - 80 
    
    # Draw Title
    draw.text((text_x, text_y_start), title_text, font=font_main, fill=TEXT_COLOR_MAIN, spacing=15)
    
    # Draw Subtitle (below title)
    # Estimate height of title
    bbox = draw.multiline_textbbox((text_x, text_y_start), title_text, font=font_main, spacing=15)
    title_height = bbox[3] - bbox[1]
    
    subtitle_y = text_y_start + title_height + 30
    draw.text((text_x, subtitle_y), subtitle_text, font=font_sub, fill=TEXT_COLOR_SUB, spacing=10)
    
    # Save
    img.save(OUTPUT_PATH)
    print(f"✅ Social banner saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    create_banner()
