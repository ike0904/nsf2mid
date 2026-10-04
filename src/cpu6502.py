"""
6502 CPU core (NES 2A03 flavour: no decimal mode).

Memory access is delegated to a bus object that provides:
    read(addr) -> int
    write(addr, value)
The bus may also expose `cycles_ref`, but the CPU keeps its own cycle counter
in `self.cycles`; the bus can read it through the cpu reference.

Official opcodes plus the commonly used unofficial ones (LAX, SAX, DCP, ISC,
SLO, RLA, SRE, RRA, ANC, ALR, ARR, SBX, multi-byte NOPs) are implemented.
Page-crossing penalty cycles are not emulated (timing inside a frame only
needs to be approximate for register logging).
"""

# Status flag bits
C = 0x01
Z = 0x02
I = 0x04
D = 0x08
B = 0x10
U = 0x20
V = 0x40
N = 0x80


class CPU6502:
    def __init__(self, bus):
        self.bus = bus
        self.a = 0
        self.x = 0
        self.y = 0
        self.s = 0xFD
        self.p = I | U
        self.pc = 0
        self.cycles = 0
        self.jammed = False
        self._build_table()

    # ------------------------------------------------------------------ helpers
    def _rd(self, addr):
        return self.bus.read(addr)

    def _wr(self, addr, val):
        self.bus.write(addr, val)

    def _rd16(self, addr):
        return self._rd(addr) | (self._rd((addr + 1) & 0xFFFF) << 8)

    def _rd16_bug(self, addr):
        # JMP (ind) page wrap bug
        hi = (addr & 0xFF00) | ((addr + 1) & 0x00FF)
        return self._rd(addr) | (self._rd(hi) << 8)

    def push(self, val):
        self._wr(0x100 | self.s, val & 0xFF)
        self.s = (self.s - 1) & 0xFF

    def pull(self):
        self.s = (self.s + 1) & 0xFF
        return self._rd(0x100 | self.s)

    def _nz(self, v):
        p = self.p & ~(N | Z)
        if v == 0:
            p |= Z
        p |= v & N
        self.p = p
        return v

    # --------------------------------------------------------- addressing modes
    # Each returns an effective address (or None for implied/accumulator).
    def am_imm(self):
        a = self.pc
        self.pc = (self.pc + 1) & 0xFFFF
        return a

    def am_zp(self):
        a = self._rd(self.pc)
        self.pc = (self.pc + 1) & 0xFFFF
        return a

    def am_zpx(self):
        a = (self._rd(self.pc) + self.x) & 0xFF
        self.pc = (self.pc + 1) & 0xFFFF
        return a

    def am_zpy(self):
        a = (self._rd(self.pc) + self.y) & 0xFF
        self.pc = (self.pc + 1) & 0xFFFF
        return a

    def am_abs(self):
        a = self._rd16(self.pc)
        self.pc = (self.pc + 2) & 0xFFFF
        return a

    def am_abx(self):
        a = (self._rd16(self.pc) + self.x) & 0xFFFF
        self.pc = (self.pc + 2) & 0xFFFF
        return a

    def am_aby(self):
        a = (self._rd16(self.pc) + self.y) & 0xFFFF
        self.pc = (self.pc + 2) & 0xFFFF
        return a

    def am_izx(self):
        z = (self._rd(self.pc) + self.x) & 0xFF
        self.pc = (self.pc + 1) & 0xFFFF
        return self._rd(z) | (self._rd((z + 1) & 0xFF) << 8)

    def am_izy(self):
        z = self._rd(self.pc)
        self.pc = (self.pc + 1) & 0xFFFF
        base = self._rd(z) | (self._rd((z + 1) & 0xFF) << 8)
        return (base + self.y) & 0xFFFF

    def am_ind(self):
        a = self._rd16(self.pc)
        self.pc = (self.pc + 2) & 0xFFFF
        return self._rd16_bug(a)

    def am_rel(self):
        off = self._rd(self.pc)
        self.pc = (self.pc + 1) & 0xFFFF
        if off & 0x80:
            off -= 0x100
        return (self.pc + off) & 0xFFFF

    def am_imp(self):
        return None

    # -------------------------------------------------------------- operations
    def op_lda(self, a): self.a = self._nz(self._rd(a))
    def op_ldx(self, a): self.x = self._nz(self._rd(a))
    def op_ldy(self, a): self.y = self._nz(self._rd(a))
    def op_sta(self, a): self._wr(a, self.a)
    def op_stx(self, a): self._wr(a, self.x)
    def op_sty(self, a): self._wr(a, self.y)

    def op_tax(self, a): self.x = self._nz(self.a)
    def op_tay(self, a): self.y = self._nz(self.a)
    def op_txa(self, a): self.a = self._nz(self.x)
    def op_tya(self, a): self.a = self._nz(self.y)
    def op_tsx(self, a): self.x = self._nz(self.s)
    def op_txs(self, a): self.s = self.x

    def op_pha(self, a): self.push(self.a)
    def op_php(self, a): self.push(self.p | B | U)
    def op_pla(self, a): self.a = self._nz(self.pull())
    def op_plp(self, a): self.p = (self.pull() & ~B) | U

    def _adc(self, v):
        s = self.a + v + (self.p & C)
        p = self.p & ~(C | V)
        if s > 0xFF:
            p |= C
        if (~(self.a ^ v) & (self.a ^ s)) & 0x80:
            p |= V
        self.p = p
        self.a = self._nz(s & 0xFF)

    def op_adc(self, a): self._adc(self._rd(a))
    def op_sbc(self, a): self._adc(self._rd(a) ^ 0xFF)

    def op_and(self, a): self.a = self._nz(self.a & self._rd(a))
    def op_ora(self, a): self.a = self._nz(self.a | self._rd(a))
    def op_eor(self, a): self.a = self._nz(self.a ^ self._rd(a))

    def _cmp(self, r, v):
        d = r - v
        self.p = (self.p & ~C) | (C if d >= 0 else 0)
        self._nz(d & 0xFF)

    def op_cmp(self, a): self._cmp(self.a, self._rd(a))
    def op_cpx(self, a): self._cmp(self.x, self._rd(a))
    def op_cpy(self, a): self._cmp(self.y, self._rd(a))

    def op_bit(self, a):
        v = self._rd(a)
        p = self.p & ~(N | V | Z)
        p |= v & (N | V)
        if (self.a & v) == 0:
            p |= Z
        self.p = p

    def op_inc(self, a):
        v = (self._rd(a) + 1) & 0xFF
        self._wr(a, v)
        self._nz(v)

    def op_dec(self, a):
        v = (self._rd(a) - 1) & 0xFF
        self._wr(a, v)
        self._nz(v)

    def op_inx(self, a): self.x = self._nz((self.x + 1) & 0xFF)
    def op_iny(self, a): self.y = self._nz((self.y + 1) & 0xFF)
    def op_dex(self, a): self.x = self._nz((self.x - 1) & 0xFF)
    def op_dey(self, a): self.y = self._nz((self.y - 1) & 0xFF)

    # shifts: memory versions
    def _asl(self, v):
        self.p = (self.p & ~C) | ((v >> 7) & 1)
        return self._nz((v << 1) & 0xFF)

    def _lsr(self, v):
        self.p = (self.p & ~C) | (v & 1)
        return self._nz(v >> 1)

    def _rol(self, v):
        c = self.p & C
        self.p = (self.p & ~C) | ((v >> 7) & 1)
        return self._nz(((v << 1) | c) & 0xFF)

    def _ror(self, v):
        c = self.p & C
        self.p = (self.p & ~C) | (v & 1)
        return self._nz((v >> 1) | (c << 7))

    def op_asl(self, a):
        if a is None:
            self.a = self._asl(self.a)
        else:
            self._wr(a, self._asl(self._rd(a)))

    def op_lsr(self, a):
        if a is None:
            self.a = self._lsr(self.a)
        else:
            self._wr(a, self._lsr(self._rd(a)))

    def op_rol(self, a):
        if a is None:
            self.a = self._rol(self.a)
        else:
            self._wr(a, self._rol(self._rd(a)))

    def op_ror(self, a):
        if a is None:
            self.a = self._ror(self.a)
        else:
            self._wr(a, self._ror(self._rd(a)))

    def op_jmp(self, a): self.pc = a

    def op_jsr(self, a):
        ret = (self.pc - 1) & 0xFFFF
        self.push(ret >> 8)
        self.push(ret & 0xFF)
        self.pc = a

    def op_rts(self, a):
        lo = self.pull()
        hi = self.pull()
        self.pc = (((hi << 8) | lo) + 1) & 0xFFFF

    def op_rti(self, a):
        self.p = (self.pull() & ~B) | U
        lo = self.pull()
        hi = self.pull()
        self.pc = (hi << 8) | lo

    def op_brk(self, a):
        self.pc = (self.pc + 1) & 0xFFFF
        self.push(self.pc >> 8)
        self.push(self.pc & 0xFF)
        self.push(self.p | B | U)
        self.p |= I
        self.pc = self._rd16(0xFFFE)

    def _branch(self, a, cond):
        if cond:
            self.cycles += 1
            self.pc = a

    def op_bpl(self, a): self._branch(a, not (self.p & N))
    def op_bmi(self, a): self._branch(a, self.p & N)
    def op_bvc(self, a): self._branch(a, not (self.p & V))
    def op_bvs(self, a): self._branch(a, self.p & V)
    def op_bcc(self, a): self._branch(a, not (self.p & C))
    def op_bcs(self, a): self._branch(a, self.p & C)
    def op_bne(self, a): self._branch(a, not (self.p & Z))
    def op_beq(self, a): self._branch(a, self.p & Z)

    def op_clc(self, a): self.p &= ~C
    def op_sec(self, a): self.p |= C
    def op_cli(self, a): self.p &= ~I
    def op_sei(self, a): self.p |= I
    def op_cld(self, a): self.p &= ~D
    def op_sed(self, a): self.p |= D
    def op_clv(self, a): self.p &= ~V

    def op_nop(self, a):
        pass

    def op_nopr(self, a):
        # unofficial NOP with operand read
        if a is not None:
            self._rd(a)

    # ---- unofficial
    def op_lax(self, a):
        self.a = self.x = self._nz(self._rd(a))

    def op_sax(self, a):
        self._wr(a, self.a & self.x)

    def op_dcp(self, a):
        v = (self._rd(a) - 1) & 0xFF
        self._wr(a, v)
        self._cmp(self.a, v)

    def op_isc(self, a):
        v = (self._rd(a) + 1) & 0xFF
        self._wr(a, v)
        self._adc(v ^ 0xFF)

    def op_slo(self, a):
        v = self._asl(self._rd(a))
        self._wr(a, v)
        self.a = self._nz(self.a | v)

    def op_rla(self, a):
        v = self._rol(self._rd(a))
        self._wr(a, v)
        self.a = self._nz(self.a & v)

    def op_sre(self, a):
        v = self._lsr(self._rd(a))
        self._wr(a, v)
        self.a = self._nz(self.a ^ v)

    def op_rra(self, a):
        v = self._ror(self._rd(a))
        self._wr(a, v)
        self._adc(v)

    def op_anc(self, a):
        self.a = self._nz(self.a & self._rd(a))
        self.p = (self.p & ~C) | ((self.a >> 7) & 1)

    def op_alr(self, a):
        self.a = self._lsr(self.a & self._rd(a))

    def op_arr(self, a):
        v = self.a & self._rd(a)
        r = ((v >> 1) | ((self.p & C) << 7)) & 0xFF
        self._nz(r)
        p = self.p & ~(C | V)
        if r & 0x40:
            p |= C
        if ((r >> 6) ^ (r >> 5)) & 1:
            p |= V
        self.p = p
        self.a = r

    def op_sbx(self, a):
        v = (self.a & self.x) - self._rd(a)
        self.p = (self.p & ~C) | (C if v >= 0 else 0)
        self.x = self._nz(v & 0xFF)

    def op_jam(self, a):
        self.jammed = True
        self.pc = (self.pc - 1) & 0xFFFF

    # -------------------------------------------------------------- op table
    def _build_table(self):
        t = [None] * 256

        def d(op, mode, cyc, codes):
            for code in codes:
                t[code] = (getattr(self, "op_" + op), getattr(self, "am_" + mode), cyc)

        # (opcode, mode, cycles)
        spec = [
            ("adc", [(0x69, "imm", 2), (0x65, "zp", 3), (0x75, "zpx", 4), (0x6D, "abs", 4),
                     (0x7D, "abx", 4), (0x79, "aby", 4), (0x61, "izx", 6), (0x71, "izy", 5)]),
            ("and", [(0x29, "imm", 2), (0x25, "zp", 3), (0x35, "zpx", 4), (0x2D, "abs", 4),
                     (0x3D, "abx", 4), (0x39, "aby", 4), (0x21, "izx", 6), (0x31, "izy", 5)]),
            ("asl", [(0x0A, "imp", 2), (0x06, "zp", 5), (0x16, "zpx", 6), (0x0E, "abs", 6), (0x1E, "abx", 7)]),
            ("bcc", [(0x90, "rel", 2)]), ("bcs", [(0xB0, "rel", 2)]), ("beq", [(0xF0, "rel", 2)]),
            ("bmi", [(0x30, "rel", 2)]), ("bne", [(0xD0, "rel", 2)]), ("bpl", [(0x10, "rel", 2)]),
            ("bvc", [(0x50, "rel", 2)]), ("bvs", [(0x70, "rel", 2)]),
            ("bit", [(0x24, "zp", 3), (0x2C, "abs", 4)]),
            ("brk", [(0x00, "imp", 7)]),
            ("clc", [(0x18, "imp", 2)]), ("cld", [(0xD8, "imp", 2)]), ("cli", [(0x58, "imp", 2)]),
            ("clv", [(0xB8, "imp", 2)]),
            ("cmp", [(0xC9, "imm", 2), (0xC5, "zp", 3), (0xD5, "zpx", 4), (0xCD, "abs", 4),
                     (0xDD, "abx", 4), (0xD9, "aby", 4), (0xC1, "izx", 6), (0xD1, "izy", 5)]),
            ("cpx", [(0xE0, "imm", 2), (0xE4, "zp", 3), (0xEC, "abs", 4)]),
            ("cpy", [(0xC0, "imm", 2), (0xC4, "zp", 3), (0xCC, "abs", 4)]),
            ("dec", [(0xC6, "zp", 5), (0xD6, "zpx", 6), (0xCE, "abs", 6), (0xDE, "abx", 7)]),
            ("dex", [(0xCA, "imp", 2)]), ("dey", [(0x88, "imp", 2)]),
            ("eor", [(0x49, "imm", 2), (0x45, "zp", 3), (0x55, "zpx", 4), (0x4D, "abs", 4),
                     (0x5D, "abx", 4), (0x59, "aby", 4), (0x41, "izx", 6), (0x51, "izy", 5)]),
            ("inc", [(0xE6, "zp", 5), (0xF6, "zpx", 6), (0xEE, "abs", 6), (0xFE, "abx", 7)]),
            ("inx", [(0xE8, "imp", 2)]), ("iny", [(0xC8, "imp", 2)]),
            ("jmp", [(0x4C, "abs", 3), (0x6C, "ind", 5)]),
            ("jsr", [(0x20, "abs", 6)]),
            ("lda", [(0xA9, "imm", 2), (0xA5, "zp", 3), (0xB5, "zpx", 4), (0xAD, "abs", 4),
                     (0xBD, "abx", 4), (0xB9, "aby", 4), (0xA1, "izx", 6), (0xB1, "izy", 5)]),
            ("ldx", [(0xA2, "imm", 2), (0xA6, "zp", 3), (0xB6, "zpy", 4), (0xAE, "abs", 4), (0xBE, "aby", 4)]),
            ("ldy", [(0xA0, "imm", 2), (0xA4, "zp", 3), (0xB4, "zpx", 4), (0xAC, "abs", 4), (0xBC, "abx", 4)]),
            ("lsr", [(0x4A, "imp", 2), (0x46, "zp", 5), (0x56, "zpx", 6), (0x4E, "abs", 6), (0x5E, "abx", 7)]),
            ("nop", [(0xEA, "imp", 2)]),
            ("ora", [(0x09, "imm", 2), (0x05, "zp", 3), (0x15, "zpx", 4), (0x0D, "abs", 4),
                     (0x1D, "abx", 4), (0x19, "aby", 4), (0x01, "izx", 6), (0x11, "izy", 5)]),
            ("pha", [(0x48, "imp", 3)]), ("php", [(0x08, "imp", 3)]),
            ("pla", [(0x68, "imp", 4)]), ("plp", [(0x28, "imp", 4)]),
            ("rol", [(0x2A, "imp", 2), (0x26, "zp", 5), (0x36, "zpx", 6), (0x2E, "abs", 6), (0x3E, "abx", 7)]),
            ("ror", [(0x6A, "imp", 2), (0x66, "zp", 5), (0x76, "zpx", 6), (0x6E, "abs", 6), (0x7E, "abx", 7)]),
            ("rti", [(0x40, "imp", 6)]), ("rts", [(0x60, "imp", 6)]),
            ("sbc", [(0xE9, "imm", 2), (0xE5, "zp", 3), (0xF5, "zpx", 4), (0xED, "abs", 4),
                     (0xFD, "abx", 4), (0xF9, "aby", 4), (0xE1, "izx", 6), (0xF1, "izy", 5),
                     (0xEB, "imm", 2)]),
            ("sec", [(0x38, "imp", 2)]), ("sed", [(0xF8, "imp", 2)]), ("sei", [(0x78, "imp", 2)]),
            ("sta", [(0x85, "zp", 3), (0x95, "zpx", 4), (0x8D, "abs", 4), (0x9D, "abx", 5),
                     (0x99, "aby", 5), (0x81, "izx", 6), (0x91, "izy", 6)]),
            ("stx", [(0x86, "zp", 3), (0x96, "zpy", 4), (0x8E, "abs", 4)]),
            ("sty", [(0x84, "zp", 3), (0x94, "zpx", 4), (0x8C, "abs", 4)]),
            ("tax", [(0xAA, "imp", 2)]), ("tay", [(0xA8, "imp", 2)]), ("tsx", [(0xBA, "imp", 2)]),
            ("txa", [(0x8A, "imp", 2)]), ("txs", [(0x9A, "imp", 2)]), ("tya", [(0x98, "imp", 2)]),
            # unofficial
            ("lax", [(0xA7, "zp", 3), (0xB7, "zpy", 4), (0xAF, "abs", 4), (0xBF, "aby", 4),
                     (0xA3, "izx", 6), (0xB3, "izy", 5), (0xAB, "imm", 2)]),
            ("sax", [(0x87, "zp", 3), (0x97, "zpy", 4), (0x8F, "abs", 4), (0x83, "izx", 6)]),
            ("dcp", [(0xC7, "zp", 5), (0xD7, "zpx", 6), (0xCF, "abs", 6), (0xDF, "abx", 7),
                     (0xDB, "aby", 7), (0xC3, "izx", 8), (0xD3, "izy", 8)]),
            ("isc", [(0xE7, "zp", 5), (0xF7, "zpx", 6), (0xEF, "abs", 6), (0xFF, "abx", 7),
                     (0xFB, "aby", 7), (0xE3, "izx", 8), (0xF3, "izy", 8)]),
            ("slo", [(0x07, "zp", 5), (0x17, "zpx", 6), (0x0F, "abs", 6), (0x1F, "abx", 7),
                     (0x1B, "aby", 7), (0x03, "izx", 8), (0x13, "izy", 8)]),
            ("rla", [(0x27, "zp", 5), (0x37, "zpx", 6), (0x2F, "abs", 6), (0x3F, "abx", 7),
                     (0x3B, "aby", 7), (0x23, "izx", 8), (0x33, "izy", 8)]),
            ("sre", [(0x47, "zp", 5), (0x57, "zpx", 6), (0x4F, "abs", 6), (0x5F, "abx", 7),
                     (0x5B, "aby", 7), (0x43, "izx", 8), (0x53, "izy", 8)]),
            ("rra", [(0x67, "zp", 5), (0x77, "zpx", 6), (0x6F, "abs", 6), (0x7F, "abx", 7),
                     (0x7B, "aby", 7), (0x63, "izx", 8), (0x73, "izy", 8)]),
            ("anc", [(0x0B, "imm", 2), (0x2B, "imm", 2)]),
            ("alr", [(0x4B, "imm", 2)]),
            ("arr", [(0x6B, "imm", 2)]),
            ("sbx", [(0xCB, "imm", 2)]),
            ("nop", [(0x1A, "imp", 2), (0x3A, "imp", 2), (0x5A, "imp", 2), (0x7A, "imp", 2),
                     (0xDA, "imp", 2), (0xFA, "imp", 2)]),
            ("nopr", [(0x80, "imm", 2), (0x82, "imm", 2), (0x89, "imm", 2), (0xC2, "imm", 2),
                      (0xE2, "imm", 2), (0x04, "zp", 3), (0x44, "zp", 3), (0x64, "zp", 3),
                      (0x14, "zpx", 4), (0x34, "zpx", 4), (0x54, "zpx", 4), (0x74, "zpx", 4),
                      (0xD4, "zpx", 4), (0xF4, "zpx", 4), (0x0C, "abs", 4), (0x1C, "abx", 4),
                      (0x3C, "abx", 4), (0x5C, "abx", 4), (0x7C, "abx", 4), (0xDC, "abx", 4),
                      (0xFC, "abx", 4)]),
            ("jam", [(0x02, "imp", 2), (0x12, "imp", 2), (0x22, "imp", 2), (0x32, "imp", 2),
                     (0x42, "imp", 2), (0x52, "imp", 2), (0x62, "imp", 2), (0x72, "imp", 2),
                     (0x92, "imp", 2), (0xB2, "imp", 2), (0xD2, "imp", 2), (0xF2, "imp", 2)]),
        ]
        for op, entries in spec:
            for code, mode, cyc in entries:
                d(op, mode, cyc, [code])
        # Remaining rarely-used unstable opcodes (SHA/SHX/SHY/TAS/LAS/XAA) -> treat as NOP w/ operand
        rest = {0x93: "izy", 0x9F: "aby", 0x9E: "aby", 0x9C: "abx", 0x9B: "aby", 0xBB: "aby", 0x8B: "imm"}
        for code, mode in rest.items():
            if t[code] is None:
                d("nopr", mode, 5, [code])
        for i in range(256):
            if t[i] is None:
                d("nop", "imp", 2, [i])
        self.table = t

    # -------------------------------------------------------------- execution
    def step(self):
        op = self._rd(self.pc)
        self.pc = (self.pc + 1) & 0xFFFF
        fn, mode, cyc = self.table[op]
        self.cycles += cyc
        fn(mode())
        return cyc
