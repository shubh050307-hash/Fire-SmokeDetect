/**
 * Enhanced Fire & Smoke Detection Overlay System
 * Stylish, animated, heat-mapped visualization with particle effects
 */

class DetectionOverlay {
  constructor(canvas, config = {}) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')
    this.particles = []
    this.animationFrame = 0
    this.config = {
      glowIntensity: 0.8,
      particleCount: 8,
      pulseSpeed: 2,
      ...config,
    }
  }

  /**
   * Main render function - call this on each frame
   */
  render(vw, vh, frame) {
    const ctx = this.ctx
    ctx.save()

    this.animationFrame += this.config.pulseSpeed

    // Always render system status (even when scanning)
    if (frame?.system === 'scanning' || (!frame?.fire && !frame?.smoke)) {
      this.renderSystemStatus(ctx, vw, vh, frame)
      ctx.restore()
      return
    }

    // Render detection overlays only if fire or smoke detected
    if (frame?.fire || frame?.smoke) {
      this.renderSmokeZones(ctx, vw, vh, frame)
      this.renderFireZones(ctx, vw, vh, frame)
      this.renderParticles(ctx, vw, vh)
      this.renderLabels(ctx, vw, vh, frame)
    }

    this.renderSystemStatus(ctx, vw, vh, frame)
    ctx.restore()
  }

  /**
   * Render smoke detection zones with elegant styling
   */
  renderSmokeZones(ctx, vw, vh, frame) {
    const smoke = frame.smoke_boxes || []
    if (!smoke.length) return

    smoke.forEach((smokeBox, idx) => {
      const b = smokeBox?.bbox
      if (!b) return

      const x = (b.x - b.w / 2) * vw
      const y = (b.y - b.h / 2) * vh
      const w = b.w * vw
      const h = b.h * vh

      const pulse = Math.sin(this.animationFrame * 0.025 + idx * 0.5) * 0.08 + 0.92
      const alpha = 0.18 * pulse

      // Smoke gradient (cool blues/purples)
      const gradient = ctx.createLinearGradient(x, y, x, y + h)
      gradient.addColorStop(0, `rgba(129, 140, 248, ${alpha * 0.6})`)
      gradient.addColorStop(0.5, `rgba(139, 92, 246, ${alpha * 0.4})`)
      gradient.addColorStop(1, `rgba(99, 102, 241, ${alpha * 0.3})`)

      this.drawRoundedRect(ctx, x, y, w, h, 8)
      ctx.fillStyle = gradient
      ctx.fill()

      ctx.strokeStyle = `rgba(165, 180, 252, ${0.85 * pulse})`
      ctx.lineWidth = 2
      ctx.setLineDash([10, 7])
      ctx.lineDashOffset = -(this.animationFrame * 0.25)
      ctx.stroke()
      ctx.setLineDash([])
    })
  }

  /**
   * Render fire detection zones with heat mapping and glow
   */
  renderFireZones(ctx, vw, vh, frame) {
    const masks = frame.fire_masks || []
    const fireBoxes = frame.fire_boxes || []
    const fireArea = Number(frame.fire_segment_area_pixels || 0)

    // Calculate intensity (0-1) based on fire area
    const maxArea = 120000
    const intensity = Math.min(fireArea / maxArea, 1)

    if (masks.length > 0) {
      // Render polygonal masks (for segmentation)
      this.renderFireMasks(ctx, vw, vh, masks, intensity)
    } else if (fireBoxes.length > 0) {
      // Render bounding boxes
      this.renderFireBoxes(ctx, vw, vh, fireBoxes, intensity)
    } else if (frame.bbox && frame.fire) {
      // Fallback to frame bbox
      const b = frame.bbox
      const x = (b.x - b.w / 2) * vw
      const y = (b.y - b.h / 2) * vh
      const w = b.w * vw
      const h = b.h * vh
      this.renderFireBox(ctx, x, y, w, h, intensity)
    }

    // Spawn particles in fire zones
    if (intensity > 0.2) {
      this.spawnFireParticles(vw, vh, frame, intensity)
    }
  }

  /**
   * Render fire masks with smooth polygons and glow effects
   */
  renderFireMasks(ctx, vw, vh, masks, intensity) {
    masks.forEach((poly) => {
      if (!poly || poly.length < 3) return

      const color = this.getHeatColor(intensity)

      const bounds = this.getPolyBounds(poly, vw, vh)
      const gradient = ctx.createRadialGradient(
        bounds.cx,
        bounds.cy,
        0,
        bounds.cx,
        bounds.cy,
        Math.max(bounds.w, bounds.h),
      )
      gradient.addColorStop(0, `rgba(${color.r}, ${color.g}, ${color.b}, ${0.35 * intensity})`)
      gradient.addColorStop(0.7, `rgba(${color.r}, ${color.g}, ${color.b}, ${0.1 * intensity})`)
      gradient.addColorStop(1, `rgba(${color.r}, ${color.g}, ${color.b}, 0)`)

      ctx.fillStyle = gradient
      ctx.beginPath()
      ctx.moveTo(poly[0].x * vw, poly[0].y * vh)
      for (let j = 1; j < poly.length; j++) {
        ctx.lineTo(poly[j].x * vw, poly[j].y * vh)
      }
      ctx.closePath()
      ctx.fill()

      ctx.shadowColor = `rgba(${color.r}, ${color.g}, ${color.b}, ${0.28 * intensity})`
      ctx.shadowBlur = 8
      ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.95)`
      ctx.lineWidth = 2
      ctx.stroke()
      ctx.shadowBlur = 0
    })
  }

  /**
   * Render fire bounding boxes with heat-based styling
   */
  renderFireBoxes(ctx, vw, vh, fireBoxes, intensity) {
    fireBoxes.forEach((box) => {
      const b = box?.bbox
      if (!b) return
      const x = (b.x - b.w / 2) * vw
      const y = (b.y - b.h / 2) * vh
      const w = b.w * vw
      const h = b.h * vh
      this.renderFireBox(ctx, x, y, w, h, intensity)
    })
  }

  /**
   * Render a single fire box with glow and animations
   */
  renderFireBox(ctx, x, y, w, h, intensity) {
    const color = this.getHeatColor(intensity)

    const pulse = Math.sin(this.animationFrame * 0.035) * 0.08 + 0.92

    const gradient = ctx.createLinearGradient(x, y, x, y + h)
    gradient.addColorStop(0, `rgba(${color.r}, ${color.g}, ${color.b}, ${0.18 * intensity})`)
    gradient.addColorStop(1, `rgba(${color.r}, ${color.g}, ${color.b}, ${0.08 * intensity})`)

    ctx.fillStyle = gradient
    this.drawRoundedRect(ctx, x, y, w, h, 6)
    ctx.fill()

    ctx.shadowColor = `rgba(${color.r}, ${color.g}, ${color.b}, ${0.25 * intensity})`
    ctx.shadowBlur = 8
    ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${0.95 * pulse})`
    ctx.lineWidth = 2
    this.drawRoundedRect(ctx, x, y, w, h, 6)
    ctx.stroke()
    ctx.shadowBlur = 0

    this.drawCornerAccents(ctx, x, y, w, h, color, intensity * pulse)
  }

  /**
   * Render animated particles (fire sparks)
   */
  renderParticles(ctx) {
    this.particles = this.particles.filter((p) => p.life > 0)

    this.particles.forEach((p) => {
      p.life -= 0.02
      p.x += p.vx
      p.y += p.vy
      p.vy += 0.1 // gravity

      const alpha = Math.max(0, p.life)
      const size = p.size * alpha

      // Particle glow
      ctx.shadowColor = `rgba(${p.color.r}, ${p.color.g}, ${p.color.b}, ${alpha})`
      ctx.shadowBlur = size * 2

      ctx.fillStyle = `rgba(${p.color.r}, ${p.color.g}, ${p.color.b}, ${alpha})`
      ctx.beginPath()
      ctx.arc(p.x, p.y, size / 2, 0, Math.PI * 2)
      ctx.fill()
    })

    ctx.shadowColor = 'transparent'
  }

  /**
   * Spawn fire particles in detection zones
   */
  spawnFireParticles(vw, vh, frame, intensity) {
    const count = Math.ceil(this.config.particleCount * intensity)
    const fireBoxes = frame.fire_boxes || []
    const masks = frame.fire_masks || []

    let spawnAreas = []

    if (masks.length > 0) {
      spawnAreas = masks.map((poly) => {
        const bounds = this.getPolyBounds(poly, vw, vh)
        return {
          x: bounds.cx,
          y: bounds.cy,
          w: bounds.w,
          h: bounds.h,
        }
      })
    } else if (fireBoxes.length > 0) {
      spawnAreas = fireBoxes.map((box) => {
        const b = box?.bbox
        if (!b) return null
        return {
          x: (b.x - b.w / 2) * vw,
          y: (b.y - b.h / 2) * vh,
          w: b.w * vw,
          h: b.h * vh,
        }
      })
    } else if (frame.bbox && frame.fire) {
      const b = frame.bbox
      spawnAreas = [
        {
          x: (b.x - b.w / 2) * vw,
          y: (b.y - b.h / 2) * vh,
          w: b.w * vw,
          h: b.h * vh,
        },
      ]
    }

    spawnAreas.forEach((area) => {
      if (!area) return
      for (let i = 0; i < count; i++) {
        const color = this.getHeatColor(Math.random() * intensity)
        this.particles.push({
          x: area.x + Math.random() * area.w,
          y: area.y + Math.random() * area.h,
          vx: (Math.random() - 0.5) * 1.5,
          vy: (Math.random() - 1) * 0.8,
          size: Math.random() * 4 + 2,
          life: 1,
          color,
        })
      }
    })
  }

  /**
   * Render information labels with statistics
   */
  renderLabels(ctx, vw, vh, frame) {
    const labels = []

    if (frame.fire) {
      const fireArea = Number(frame.fire_segment_area_pixels || 0)
      const intensity = Math.min(fireArea / 120000, 1)
      const anchor = this.getFireAnchor(frame, vw, vh) || { x: 24, y: 24 }

      labels.push({
        title: 'FIRE',
        value: `${Math.round(fireArea).toLocaleString()} px`,
        x: anchor.x,
        y: anchor.y,
        color: this.getHeatColor(intensity),
      })
    }

    if (frame.smoke) {
      const anchor = this.getSmokeAnchor(frame, vw, vh) || { x: 24, y: frame.fire ? 76 : 24 }
      labels.push({
        title: 'SMOKE',
        value: 'zone identified',
        x: anchor.x,
        y: anchor.y,
        color: { r: 129, g: 140, b: 248 },
      })
    }

    labels.forEach((label) => {
      this.drawLabel(ctx, label, vw, vh)
    })
  }

  /**
   * Draw a styled label with background and shadow
   */
  drawLabel(ctx, label, vw, vh) {
    const color = label.color
    const title = label.title || ''
    const value = label.value || ''

    ctx.font = '800 12px Inter, system-ui, sans-serif'
    const titleWidth = ctx.measureText(title).width
    ctx.font = '600 11px Inter, system-ui, sans-serif'
    const valueWidth = ctx.measureText(value).width

    const paddingX = 12
    const bgW = Math.max(92, titleWidth + valueWidth + paddingX * 2 + 18)
    const bgH = 34
    const bgX = Math.min(Math.max(label.x, 10), Math.max(10, vw - bgW - 10))
    const bgY = Math.min(Math.max(label.y - bgH - 8, 10), Math.max(10, vh - bgH - 10))

    ctx.shadowColor = 'rgba(0, 0, 0, 0.35)'
    ctx.shadowBlur = 8
    ctx.fillStyle = 'rgba(7, 12, 22, 0.86)'
    this.drawRoundedRect(ctx, bgX, bgY, bgW, bgH, 7)
    ctx.fill()

    ctx.shadowBlur = 0
    ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.9)`
    ctx.lineWidth = 1
    this.drawRoundedRect(ctx, bgX, bgY, bgW, bgH, 7)
    ctx.stroke()

    ctx.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`
    ctx.font = '800 12px Inter, system-ui, sans-serif'
    ctx.fillText(title, bgX + paddingX, bgY + 21)

    ctx.fillStyle = 'rgba(226, 232, 240, 0.9)'
    ctx.font = '600 11px Inter, system-ui, sans-serif'
    ctx.fillText(value, bgX + paddingX + titleWidth + 14, bgY + 21)
  }

  /**
   * Draw corner accents for tactical look
   */
  drawCornerAccents(ctx, x, y, w, h, color, intensity) {
    const accentLen = Math.min(w, h) * 0.15
    const accentWidth = 2

    ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${0.8 * intensity})`
    ctx.lineWidth = accentWidth

    // Top-left
    ctx.beginPath()
    ctx.moveTo(x, y + accentLen)
    ctx.lineTo(x, y)
    ctx.lineTo(x + accentLen, y)
    ctx.stroke()

    // Top-right
    ctx.beginPath()
    ctx.moveTo(x + w - accentLen, y)
    ctx.lineTo(x + w, y)
    ctx.lineTo(x + w, y + accentLen)
    ctx.stroke()

    // Bottom-left
    ctx.beginPath()
    ctx.moveTo(x, y + h - accentLen)
    ctx.lineTo(x, y + h)
    ctx.lineTo(x + accentLen, y + h)
    ctx.stroke()

    // Bottom-right
    ctx.beginPath()
    ctx.moveTo(x + w - accentLen, y + h)
    ctx.lineTo(x + w, y + h)
    ctx.lineTo(x + w, y + h - accentLen)
    ctx.stroke()
  }

  /**
   * Get color based on heat intensity (cool to hot gradient)
   */
  getHeatColor(intensity) {
    // Cool (blue) -> Warm (orange/red) gradient
    if (intensity < 0.3) {
      // Blue
      return { r: 34, g: 211, b: 238 }
    } else if (intensity < 0.6) {
      // Yellow
      return { r: 251, g: 191, b: 36 }
    } else {
      // Red/Orange
      return { r: 239, g: 68, b: 68 }
    }
  }

  /**
   * Draw rounded rectangle
   */
  drawRoundedRect(ctx, x, y, w, h, radius) {
    ctx.beginPath()
    ctx.moveTo(x + radius, y)
    ctx.lineTo(x + w - radius, y)
    ctx.quadraticCurveTo(x + w, y, x + w, y + radius)
    ctx.lineTo(x + w, y + h - radius)
    ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h)
    ctx.lineTo(x + radius, y + h)
    ctx.quadraticCurveTo(x, y + h, x, y + h - radius)
    ctx.lineTo(x, y + radius)
    ctx.quadraticCurveTo(x, y, x + radius, y)
    ctx.closePath()
  }

  /**
   * Get bounding box of polygon
   */
  getPolyBounds(poly, vw, vh) {
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity

    poly.forEach((p) => {
      const x = p.x * vw
      const y = p.y * vh
      minX = Math.min(minX, x)
      maxX = Math.max(maxX, x)
      minY = Math.min(minY, y)
      maxY = Math.max(maxY, y)
    })

    return {
      x: minX,
      y: minY,
      cx: (minX + maxX) / 2,
      cy: (minY + maxY) / 2,
      w: maxX - minX,
      h: maxY - minY,
    }
  }

  getFireAnchor(frame, vw, vh) {
    const masks = frame.fire_masks || []
    if (masks.length) {
      const bounds = this.getPolyBounds(masks[0], vw, vh)
      return { x: bounds.x, y: bounds.y }
    }

    const box = frame.fire_boxes?.[0]?.bbox || frame.bbox
    if (!box) return null

    return {
      x: (box.x - box.w / 2) * vw,
      y: (box.y - box.h / 2) * vh,
    }
  }

  getSmokeAnchor(frame, vw, vh) {
    const box = frame.smoke_boxes?.[0]?.bbox
    if (!box) return null

    return {
      x: (box.x - box.w / 2) * vw,
      y: (box.y - box.h / 2) * vh,
    }
  }

  /**
   * Render continuous system scanning status
   */
  renderSystemStatus(ctx, vw, vh, frame) {
    const isScanning = frame?.system === 'scanning' || (!frame.fire && !frame.smoke)
    if (!isScanning) return

    ctx.save()
    const color = { r: 6, g: 182, b: 212 } // Cyan
    const alpha = Math.sin(this.animationFrame * 0.05) * 0.2 + 0.4
    
    // Top-right status
    ctx.font = '600 10px Inter, system-ui, sans-serif'
    ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${alpha})`
    ctx.textAlign = 'right'
    ctx.fillText('SCANNING FEED', vw - 20, 28)
    
    // Reticle in center
    const cx = vw / 2
    const cy = vh / 2
    const size = 24
    ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${alpha * 0.28})`
    ctx.lineWidth = 1
    
    ctx.beginPath(); ctx.moveTo(cx - size, cy); ctx.lineTo(cx + size, cy); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(cx, cy - size); ctx.lineTo(cx, cy + size); ctx.stroke()
    
    // Scanline
    const scanY = (this.animationFrame * 1.2) % vh
    ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.06)`
    ctx.beginPath(); ctx.moveTo(0, scanY); ctx.lineTo(vw, scanY); ctx.stroke()
    
    ctx.restore()
  }

  /**
   * Clear overlay
   */
  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
  }
}

export default DetectionOverlay
